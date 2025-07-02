import logging
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from typing import Annotated

import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from haystack.dataclasses import ChatMessage
from prometheus_fastapi_instrumentator import Instrumentator

from ai4gd_momconnect_haystack.assessment_logic import (
    create_assessment_end_error_response,
    determine_task,
    get_content_from_message_data,
    response_is_required_for,
    score_assessment_question,
    validate_assessment_answer,
    validate_assessment_end_response,
)
from ai4gd_momconnect_haystack.crud import (
    calculate_and_store_assessment_result,
    delete_assessment_history_for_user,
    delete_chat_history_for_user,
    get_assessment_end_messaging_history,
    get_assessment_result,
    get_or_create_chat_history,
    save_assessment_end_message,
    save_assessment_question,
    save_chat_history,
)
from ai4gd_momconnect_haystack.doc_store import setup_document_store
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndRequest,
    AssessmentEndResponse,
    AssessmentRequest,
    AssessmentResponse,
    OnboardingRequest,
    OnboardingResponse,
    SurveyRequest,
    SurveyResponse,
)
from ai4gd_momconnect_haystack.tasks import (
    extract_anc_data_from_response,
    extract_onboarding_data_from_response,
    get_anc_survey_question,
    get_assessment_question,
    get_next_onboarding_question,
    handle_user_message,
)
from ai4gd_momconnect_haystack.utilities import (
    assessment_end_flow_map,
    load_json_and_validate,
)

from .enums import HistoryType

load_dotenv()

logger = logging.getLogger(__name__)

DATA_PATH = Path("src/ai4gd_momconnect_haystack/")

SERVICE_PERSONA_PATH = DATA_PATH / "static_content" / "service_persona.json"
SERVICE_PERSONA = load_json_and_validate(SERVICE_PERSONA_PATH, dict)
SERVICE_PERSONA_TEXT = ""
if SERVICE_PERSONA:
    if "persona" in SERVICE_PERSONA.keys():
        SERVICE_PERSONA_TEXT = SERVICE_PERSONA["persona"]


def setup_sentry():
    if dsn := environ.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=dsn, send_default_pii=True)


setup_sentry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    setup_document_store(startup=True)
    yield
    logger.info("Application shutdown...")


app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"health": "ok"}


def verify_token(authorization: Annotated[str, Header()]):
    """
    Verify the API token from the Authorization header.
    """
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "token":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication format. Expected 'Token <token>'",
        )

    if credential != environ["API_TOKEN"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )

    return credential


@app.post("/v1/onboarding")
async def onboarding(request: OnboardingRequest, token: str = Depends(verify_token)):
    user_id = request.user_id
    user_input = request.user_input
    chat_history = await get_or_create_chat_history(
        user_id=user_id, history_type=HistoryType.onboarding
    )

    if user_input:
        last_question = chat_history[-1].text if chat_history else ""
        chat_history.append(ChatMessage.from_user(text=user_input))
        intent, intent_related_response = handle_user_message(last_question, user_input)
    else:
        # For the first message we initiate, we won't have a user input, so we should
        # force the intent in this situation
        await delete_chat_history_for_user(request.user_id, HistoryType.onboarding)
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        # Also initialize the chat history with the persona in a system message
        chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA_TEXT)]
    if intent == "JOURNEY_RESPONSE" and user_input:
        user_context = extract_onboarding_data_from_response(
            user_response=user_input,
            user_context=request.user_context,
            chat_history=chat_history,
        )
    else:
        # If there is no response to the question, the context stays the same
        user_context = request.user_context
    question = get_next_onboarding_question(
        user_context=user_context, chat_history=chat_history
    )
    question_text = ""
    if question:
        question_text = question.get("contextualized_question", "")

    if question_text:
        chat_history.append(ChatMessage.from_assistant(text=question_text))
    await save_chat_history(
        user_id=request.user_id,
        messages=chat_history,
        history_type=HistoryType.onboarding,
    )

    return OnboardingResponse(
        question=question_text,
        user_context=user_context,
        intent=intent,
        intent_related_response=intent_related_response,
    )


@app.post("/v1/assessment")
async def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    if request.user_input:
        intent, intent_related_response = handle_user_message(
            request.previous_question, request.user_input
        )
    else:
        await delete_assessment_history_for_user(request.user_id, request.flow_id)
        intent, intent_related_response = "JOURNEY_RESPONSE", ""

    if intent == "JOURNEY_RESPONSE" and request.user_input:
        answer = validate_assessment_answer(
            user_response=request.user_input,
            question_number=request.question_number,
            current_flow_id=request.flow_id.value,
        )
        if answer["processed_user_response"]:
            score = score_assessment_question(
                answer["processed_user_response"],
                request.question_number,
                request.flow_id,
            )
            await save_assessment_question(
                user_id=request.user_id,
                assessment_type=request.flow_id,
                question_number=request.question_number,
                question=None,
                user_response=answer["processed_user_response"],
                score=score,
            )
            await calculate_and_store_assessment_result(
                request.user_id, request.flow_id
            )
        next_question_number = answer["next_question_number"]
    else:
        next_question_number = request.question_number

    question = await get_assessment_question(
        user_id=request.user_id,
        flow_id=request.flow_id,
        question_number=next_question_number,
        user_context=request.user_context,
    )
    contextualized_question = ""
    if question:
        contextualized_question = question["contextualized_question"]
        await save_assessment_question(
            user_id=request.user_id,
            assessment_type=request.flow_id,
            question_number=next_question_number,
            question=contextualized_question,
            user_response=None,
            score=None,
        )
    return AssessmentResponse(
        question=contextualized_question,
        next_question=next_question_number,
        intent=intent,
        intent_related_response=intent_related_response,
    )


@app.post("/v1/assessment-end")
async def assessment_end(
    request: AssessmentEndRequest, token: str = Depends(verify_token)
):
    # Initial Setup and Data Fetching
    assessment_result = await get_assessment_result(
        user_id=request.user_id,
        assessment_type=request.flow_id,
    )
    if not assessment_result:
        return create_assessment_end_error_response("Assessment results not available.")

    score_category = (
        "skipped-many"
        if assessment_result.crossed_skip_threshold
        else assessment_result.category
    )
    flow_content_list = assessment_end_flow_map[request.flow_id.value]
    flow_content_map = {item.message_nr: item for item in flow_content_list}
    messaging_history = await get_assessment_end_messaging_history(
        user_id=request.user_id, assessment_type=request.flow_id
    )

    # Determine Current State and Handle User Input
    task = ""
    previous_message_nr = 0
    if messaging_history:
        previous_message_nr = messaging_history[-1].message_number

    intent: str | None = None
    intent_related_response: str | None = None

    if not request.user_input:
        # This is the start of the assessment-end flow
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        next_message_nr = 1
    else:
        # User has responded, process their input
        if not previous_message_nr or previous_message_nr not in flow_content_map:
            return create_assessment_end_error_response(
                "User responded while this journey has either not started, or it already ended."
            )

        previous_message_data = flow_content_map[previous_message_nr]
        previous_message, previous_valid_responses = get_content_from_message_data(
            previous_message_data, score_category
        )

        if not previous_message:
            return create_assessment_end_error_response(
                "User responded without this journey having been triggered."
            )
        intent, intent_related_response = handle_user_message(
            previous_message, request.user_input
        )
        next_message_nr = previous_message_nr + 1

    # Process Intent and Determine Next Step
    next_message = ""
    if intent == "JOURNEY_RESPONSE":
        # Check if the user was responding to a question that requires validation
        if request.user_input and response_is_required_for(
            request.flow_id.value, next_message_nr - 1
        ):
            if not previous_valid_responses:
                return create_assessment_end_error_response(
                    "Valid responses not found in the previous message's data."
                )

            validation_result = validate_assessment_end_response(
                previous_message=previous_message,
                previous_message_nr=previous_message_nr,
                previous_message_valid_responses=previous_valid_responses,
                user_response=request.user_input,
            )

            if validation_result["next_message_number"] == previous_message_nr:
                # Validation failed, repeat the previous question
                next_message_nr = previous_message_nr
                next_message_data = flow_content_map[next_message_nr]
                next_message, _ = get_content_from_message_data(
                    next_message_data, score_category
                )
            else:
                # Validation succeeded
                processed_response = validation_result["processed_user_response"]
                await save_assessment_end_message(
                    request.user_id,
                    request.flow_id,
                    previous_message_nr,
                    processed_response,
                )
                task = determine_task(
                    request.flow_id.value,
                    previous_message_nr,
                    score_category,
                    processed_response,
                )
        elif request.user_input:
            # User responded, but it was to a final message that doesn't need validation. End of flow.
            next_message = ""

    # If we fall through to here OR it's the first message, we need to find the next message to send.
    if not next_message and next_message_nr in flow_content_map:
        next_message_data = flow_content_map[next_message_nr]
        next_message, _ = get_content_from_message_data(
            next_message_data, score_category
        )

        # If this is a new question being sent, save a placeholder for it
        # This covers the initial message and moving to the next valid step
        if next_message and (
            not previous_message_nr or next_message_nr > previous_message_nr
        ):
            await save_assessment_end_message(
                request.user_id, request.flow_id, next_message_nr, ""
            )
    elif intent != "JOURNEY_RESPONSE":
        # User's intent was not to continue the journey (e.g., they asked a question).
        # Re-send the last message.
        if previous_message_nr in flow_content_map:
            previous_message_data = flow_content_map[previous_message_nr]
            next_message, _ = get_content_from_message_data(
                previous_message_data, score_category
            )
        else:
            # This case is unlikely, but as a fallback, end the flow.
            next_message = ""

    return AssessmentEndResponse(
        message=next_message,
        task=task,
        intent=intent,
        intent_related_response=intent_related_response,
    )


@app.post("/v1/survey", response_model=SurveyResponse)
async def survey(request: SurveyRequest, token: str = Depends(verify_token)):
    """
    Handles the conversation flow for the ANC (Antenatal Care) survey.
    It extracts data from the user's response and determines the next question.
    """
    user_id = request.user_id
    user_input = request.user_input
    chat_history = await get_or_create_chat_history(
        user_id=user_id, history_type=request.survey_id
    )

    last_question = ""
    last_step_title = ""
    if chat_history:
        assistant_chat_messages = [
            cm for cm in chat_history if cm.role.value == "assistant"
        ]
        last_chat_message = (
            assistant_chat_messages[-1] if assistant_chat_messages else None
        )
        if last_chat_message:
            last_question = last_chat_message.text
            last_step_title = last_chat_message.meta["step_title"]
    if request.user_input:
        intent, intent_related_response = handle_user_message(last_question, user_input)
    else:
        await delete_chat_history_for_user(request.user_id, HistoryType.anc)
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA_TEXT)]
    user_context = request.user_context
    if intent == "JOURNEY_RESPONSE" and request.user_input:
        # There's only one survey type, so we can assume anc until we add more
        # First, extract data from the user's last response to update the context
        previous_context_keys = list(user_context.keys())
        user_context = extract_anc_data_from_response(
            user_response=request.user_input,
            user_context=request.user_context,
            step_title=last_step_title,
            previous_service_message=last_question,
        )
        for k, v in user_context.items():
            if k not in previous_context_keys:
                chat_history.append(ChatMessage.from_user(text=v))
                await save_chat_history(
                    user_id=request.user_id,
                    messages=chat_history,
                    history_type=request.survey_id,
                )
                # There can only be one extracted data point
                break
    # Then, get the next logical question based on the updated context
    # await get_or_create_chat_history()
    question_result = await get_anc_survey_question(
        user_id=user_id, user_context=user_context
    )
    question = ""
    survey_complete = True
    next_step = ""
    if question_result:
        question = question_result.get("contextualized_question", "")
        survey_complete = question_result.get("is_final_step", False)
        next_step = question_result.get("question_identifier", "")
    # Add the new question or a completion message to the history
    if (not question) and survey_complete:
        completion_message = "Thank you for completing the survey!"
        question = completion_message
    chat_history.append(
        ChatMessage.from_assistant(text=question, meta={"step_title": next_step})
    )
    await save_chat_history(
        user_id=request.user_id, messages=chat_history, history_type=request.survey_id
    )
    return SurveyResponse(
        question=question,
        user_context=user_context,
        survey_complete=survey_complete,
        intent=intent,
        intent_related_response=intent_related_response,
    )
