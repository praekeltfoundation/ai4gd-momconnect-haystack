import logging
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from typing import Annotated, Type

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
from ai4gd_momconnect_haystack.database import run_migrations
from ai4gd_momconnect_haystack.doc_store import (
    INTRO_MESSAGES,
    setup_document_store,
)
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndRequest,
    AssessmentEndResponse,
    AssessmentRequest,
    AssessmentResponse,
    CatchAllRequest,
    CatchAllResponse,
    OnboardingRequest,
    OnboardingResponse,
    SurveyRequest,
    SurveyResponse,
)
from ai4gd_momconnect_haystack.tasks import (
    extract_anc_data_from_response,
    extract_assessment_data_from_response,
    get_anc_survey_question,
    get_assessment_question,
    get_next_onboarding_question,
    handle_conversational_repair,
    handle_intro_response,
    handle_user_message,
    process_onboarding_step,
)
from ai4gd_momconnect_haystack.utilities import (
    FLOWS_WITH_INTRO,
    all_onboarding_questions,
    assessment_end_flow_map,
    load_json_and_validate,
    ANC_SURVEY_MAP,
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


async def _handle_consent_result(
    result: dict,
    response_model: Type[OnboardingResponse | AssessmentResponse | SurveyResponse],
    base_response_args: dict,
) -> OnboardingResponse | AssessmentResponse | SurveyResponse | None:
    action = result.get("action")
    if action == "PROCEED":
        return None
    response_args = base_response_args.copy()
    response_args.update(
        {
            "question": result["message"],
            "intent": result["intent"],
            "intent_related_response": result["intent_related_response"],
        }
    )
    if response_model is AssessmentResponse:
        response_args["next_question"] = 0 if action != "ABORT" else None
    elif response_model is SurveyResponse:
        response_args["survey_complete"] = True if action == "ABORT" else False
    return response_model(**response_args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    setup_document_store(startup=True)
    run_migrations()
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
    flow_id = "onboarding"
    chat_history = await get_or_create_chat_history(
        user_id=user_id, history_type=HistoryType.onboarding
    )

    # This block handles the very first message of a flow (no user input)
    if not user_input:
        if flow_id and flow_id in FLOWS_WITH_INTRO:
            await delete_chat_history_for_user(request.user_id, HistoryType.onboarding)
            chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA_TEXT)]
            intro_message = INTRO_MESSAGES["free_text_intro"]
            chat_history.append(ChatMessage.from_assistant(text=intro_message))
            await save_chat_history(
                user_id=user_id,
                messages=chat_history,
                history_type=HistoryType.onboarding,
            )
            return OnboardingResponse(
                question=intro_message,
                user_context=request.user_context,
                intent="SYSTEM_INTRO",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
    # This block handles the user's response to the intro message
    last_assistant_msg = next(
        (msg for msg in reversed(chat_history) if msg.role.value == "assistant"), None
    )
    is_intro_response = (
        len(chat_history) == 2 and chat_history[0].role.value == "system"
    )

    if is_intro_response:
        result = handle_intro_response(user_input=user_input, flow_id=flow_id)
        # If user chitchats on intro, trigger repair instead of aborting
        if result["intent"] == "CHITCHAT":
            rephrased_question = handle_conversational_repair(
                flow_id=flow_id,
                question_identifier=0,  # No specific question number for intro
                previous_question=last_assistant_msg.text if last_assistant_msg else "",
                invalid_input=user_input,
            )
            return OnboardingResponse(
                question=rephrased_question
                or (last_assistant_msg.text if last_assistant_msg else ""),
                user_context=request.user_context,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=request.failure_count + 1,
            )

        response = await _handle_consent_result(
            result=result,
            response_model=OnboardingResponse,
            base_response_args={
                "user_context": request.user_context,
                "results_to_save": [],
                "failure_count": 0,
            },
        )
        if response:
            return response
        # If 'response' is None, it means the user consented and we can proceed.
        chat_history.append(ChatMessage.from_user(text=user_input))

    # --- REGULAR ONBOARDING LOGIC ---
    last_question = ""
    if last_assistant_msg:
        last_question = last_assistant_msg.text
    last_question_obj = next(
        (
            q
            for q in all_onboarding_questions
            if q.content and q.content in last_question
        ),
        None,
    )
    current_question_number = (
        last_question_obj.question_number if last_question_obj else 0
    )

    # For subsequent messages, detect intent
    if not is_intro_response:
        intent, intent_related_response = handle_user_message(last_question, user_input)
        intent = intent or ""
        intent_related_response = intent_related_response or ""

    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""

    user_context = request.user_context.copy()
    results_to_save = []
    question_text = ""
    failure_count = request.failure_count

    if intent == "SKIP_QUESTION":
        failure_count = 0
        question_to_skip = next(
            (
                q
                for q in all_onboarding_questions
                if q.question_number == current_question_number
            ),
            None,
        )
        if question_to_skip and question_to_skip.collects:
            field_to_update = question_to_skip.collects
            # This is where we record the skip in the user's context
            user_context[field_to_update] = "Skip"
            results_to_save.append(field_to_update)

        next_question_data = get_next_onboarding_question(user_context=user_context)
        if next_question_data:
            question_text = next_question_data.get("contextualized_question", "")

    elif intent == "JOURNEY_RESPONSE":
        previous_context = request.user_context.copy()
        user_context, question = process_onboarding_step(
            user_input=user_input,
            current_context=previous_context,
            current_question=last_question,
        )
        if question:
            question_text = question.get("contextualized_question", "")

        diff_keys = [
            k for k in user_context if user_context.get(k) != previous_context.get(k)
        ]
        results_to_save.extend(diff_keys)

        if not diff_keys:  # Validation or extraction failed
            if request.failure_count >= 1:
                logger.warning(
                    f"Onboarding failed for question '{last_question}' twice. Skipping."
                )
                question_to_skip = next(
                    (
                        q
                        for q in all_onboarding_questions
                        if q.question_number == current_question_number
                    ),
                    None,
                )
                if question_to_skip and question_to_skip.collects:
                    field_to_update = question_to_skip.collects
                    # Record the system-initiated skip
                    user_context[field_to_update] = "Skipped - System"
                    results_to_save.append(field_to_update)
                next_question_data = get_next_onboarding_question(
                    user_context=user_context
                )
                if next_question_data:
                    question_text = next_question_data.get(
                        "contextualized_question", ""
                    )
                failure_count = 0
            else:
                # First failure, trigger conversational repair
                rephrased_question = handle_conversational_repair(
                    flow_id=flow_id,
                    question_identifier=current_question_number,
                    previous_question=last_question,
                    invalid_input=request.user_input,
                )
                return OnboardingResponse(
                    question=rephrased_question or last_question,
                    user_context=previous_context,
                    intent="REPAIR",
                    intent_related_response=None,
                    results_to_save=[],
                    failure_count=request.failure_count + 1,
                )
        else:
            failure_count = 0
    else:  # Handle other intents like CHITCHAT
        if intent == "CHITCHAT":
            rephrased_question = handle_conversational_repair(
                flow_id=flow_id,
                question_identifier=current_question_number,
                previous_question=last_question,
                invalid_input=user_input,
            )
            return OnboardingResponse(
                question=rephrased_question or last_question,
                user_context=request.user_context,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=request.failure_count + 1,
            )
        return OnboardingResponse(
            question=last_question,
            user_context=request.user_context,
            intent=intent,
            intent_related_response=intent_related_response,
            results_to_save=[],
            failure_count=failure_count,
        )

    if not is_intro_response:
        chat_history.append(ChatMessage.from_user(text=user_input))
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
        results_to_save=results_to_save,
        failure_count=failure_count,
    )


@app.post("/v1/assessment")
async def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    # --- 1. HANDLE START OF A NEW ASSESSMENT ---
    if not request.user_input:
        await delete_assessment_history_for_user(request.user_id, request.flow_id)

        # Now, check if this specific flow needs an intro message.
        if request.flow_id.value in FLOWS_WITH_INTRO:
            intro_message = (
                INTRO_MESSAGES["free_text_intro"]
                if "behaviour" in request.flow_id.value
                else INTRO_MESSAGES["multiple_choice_intro"]
            )
            return AssessmentResponse(
                question=intro_message,
                next_question=0,
                intent="SYSTEM_INTRO",
                intent_related_response=None,
                processed_answer=None,
                failure_count=0,
            )

    # --- 2. HANDLE THE USER'S RESPONSE TO THE INTRO ---
    if request.question_number == 0:
        result = handle_intro_response(
            user_input=request.user_input, flow_id=request.flow_id.value
        )
        response = await _handle_consent_result(
            result=result,
            response_model=AssessmentResponse,
            base_response_args={"processed_answer": None},
        )
        if response:
            return response

    # --- REGULAR TURN LOGIC ---
    current_question_number = (
        1 if request.question_number == 0 else request.question_number
    )
    next_question_number = current_question_number
    processed_answer = None
    intent: str | None = "JOURNEY_RESPONSE"
    intent_related_response: str | None = ""
    failure_count = request.failure_count

    if request.user_input and request.question_number != 0:
        intent, intent_related_response = handle_user_message(
            request.previous_question, request.user_input
        )
        intent = intent or ""

        if intent == "SKIP_QUESTION":
            processed_answer = "Skip"
            next_question_number = current_question_number + 1
            failure_count = 0
        elif intent == "JOURNEY_RESPONSE":
            if "behaviour" in request.flow_id.value:
                answer = extract_assessment_data_from_response(
                    user_response=request.user_input,
                    flow_id=request.flow_id.value,
                    question_number=current_question_number,
                )
            else:
                answer = validate_assessment_answer(
                    user_response=request.user_input,
                    question_number=current_question_number,
                    current_flow_id=request.flow_id.value,
                )
            processed_answer = answer.get("processed_user_response")

            if not processed_answer:
                if request.failure_count >= 1:
                    logger.warning(
                        f"Force-skipping question {current_question_number}."
                    )
                    processed_answer = "Skip"
                    next_question_number = current_question_number + 1
                    failure_count = 0
                else:
                    rephrased_question = handle_conversational_repair(
                        flow_id=request.flow_id.value,
                        question_identifier=current_question_number,
                        previous_question=request.previous_question,
                        invalid_input=request.user_input,
                    )
                    return AssessmentResponse(
                        question=rephrased_question or request.previous_question,
                        next_question=current_question_number,
                        intent="REPAIR",
                        intent_related_response=None,
                        processed_answer=None,
                        failure_count=request.failure_count + 1,
                    )
            else:
                next_question_number = answer.get(
                    "next_question_number", current_question_number + 1
                )
                failure_count = 0
        else:
            if intent == "CHITCHAT":
                rephrased_question = handle_conversational_repair(
                    flow_id=request.flow_id.value,
                    question_identifier=current_question_number,
                    previous_question=request.previous_question,
                    invalid_input=request.user_input,
                )
                return AssessmentResponse(
                    question=rephrased_question or request.previous_question,
                    next_question=current_question_number,
                    intent="REPAIR",
                    intent_related_response=None,
                    processed_answer=None,
                    failure_count=request.failure_count + 1,
                )
            return AssessmentResponse(
                question=request.previous_question,
                next_question=current_question_number,
                intent=intent,
                intent_related_response=intent_related_response,
                processed_answer=None,
                failure_count=failure_count,
            )

    if processed_answer:
        score = score_assessment_question(
            processed_answer,
            current_question_number,
            request.flow_id,
        )
        await save_assessment_question(
            user_id=request.user_id,
            assessment_type=request.flow_id,
            question_number=current_question_number,
            question=request.previous_question,
            user_response=processed_answer,
            score=score,
        )
        await calculate_and_store_assessment_result(request.user_id, request.flow_id)

    # --- FETCH AND RETURN NEXT QUESTION ---
    question = await get_assessment_question(
        user_id=request.user_id,
        flow_id=request.flow_id,
        question_number=next_question_number,
        user_context=request.user_context,
    )
    contextualized_question = ""
    if question:
        contextualized_question = question.get("contextualized_question", "")
        if contextualized_question:
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
        next_question=next_question_number if contextualized_question else None,
        intent=intent,
        intent_related_response=intent_related_response,
        processed_answer=processed_answer,
        failure_count=failure_count,
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
    task: str | None = ""
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
        task=task or "",
        intent=intent,
        intent_related_response=intent_related_response,
    )


@app.post("/v1/survey", response_model=SurveyResponse)
async def survey(request: SurveyRequest, token: str = Depends(verify_token)):
    user_id = request.user_id
    user_input = request.user_input
    flow_id = "anc-survey"
    chat_history = await get_or_create_chat_history(
        user_id=user_id, history_type=request.survey_id
    )
    user_context = request.user_context.copy()
    previous_context = request.user_context.copy()

    # --- 1. INTRO LOGIC ---
    if not user_input:
        if flow_id in FLOWS_WITH_INTRO:
            await delete_chat_history_for_user(request.user_id, HistoryType.anc)
            chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA_TEXT)]
            intro_message = INTRO_MESSAGES["free_text_intro"]
            chat_history.append(
                ChatMessage.from_assistant(
                    text=intro_message, meta={"step_title": "intro"}
                )
            )
            await save_chat_history(
                user_id=user_id, messages=chat_history, history_type=request.survey_id
            )
            return SurveyResponse(
                question=intro_message,
                user_context=user_context,
                survey_complete=False,
                intent="SYSTEM_INTRO",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )

    last_assistant_message = next(
        (msg for msg in reversed(chat_history) if msg.role.value == "assistant"), None
    )
    is_intro_response = (
        last_assistant_message
        and last_assistant_message.meta.get("step_title") == "intro"
    )

    # --- Initialize intent variables at a higher scope ---
    intent: str | None = "JOURNEY_RESPONSE"
    intent_related_response: str | None = ""

    # Append user message to history early for all subsequent turns
    if user_input:
        chat_history.append(ChatMessage.from_user(text=user_input))

    if is_intro_response:
        result = handle_intro_response(user_input=user_input, flow_id=flow_id)
        response = await _handle_consent_result(
            result=result,
            response_model=SurveyResponse,
            base_response_args={
                "user_context": user_context,
                "results_to_save": [],
                "survey_complete": False,
                "failure_count": 0,
            },
        )
        if response:
            return response
        # Consent given, fall through to get the first question

    # --- 2. CLARIFICATION LOOP (INCOMING) ---
    elif "pending_confirmation" in previous_context:
        pending_info = user_context.pop("pending_confirmation")
        step_title_to_confirm = pending_info["step_title_to_confirm"]
        potential_answer = pending_info["potential_answer"]

        if user_input.lower().strip() in ["yes", "y", "yebo", "correct"]:
            user_context[step_title_to_confirm] = potential_answer
            # Fall through to get the next question
        else:
            original_question_data = ANC_SURVEY_MAP.get(step_title_to_confirm)
            rephrased_question = handle_conversational_repair(
                flow_id=flow_id,
                question_identifier=step_title_to_confirm,
                previous_question=original_question_data.content
                if original_question_data
                else "",
                invalid_input=user_input,
            )
            return SurveyResponse(
                question=rephrased_question,
                user_context=user_context,
                survey_complete=False,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=request.failure_count + 1,
            )

    # --- 3. REGULAR SURVEY LOGIC ---
    else:
        last_question = last_assistant_message.text if last_assistant_message else ""
        last_step_title = (
            last_assistant_message.meta.get("step_title", "")
            if last_assistant_message
            else ""
        )

        last_question_data = ANC_SURVEY_MAP.get(last_step_title)
        valid_responses_for_intent = (
            last_question_data.valid_responses if last_question_data else []
        )
        intent, intent_related_response = handle_user_message(
            last_question, user_input, valid_responses_for_intent
        )

        if intent == "SKIP_QUESTION":
            user_context[last_step_title] = "Skip"
        elif intent == "JOURNEY_RESPONSE":
            user_context, action_dict = extract_anc_data_from_response(
                user_response=user_input,
                user_context=user_context,
                step_title=last_step_title,
            )

            if action_dict and action_dict.get("status") == "needs_confirmation":
                user_context["pending_confirmation"] = action_dict
                confirmation_question = action_dict["message"]
                chat_history.append(
                    ChatMessage.from_assistant(
                        text=confirmation_question, meta={"step_title": "confirmation"}
                    )
                )
                await save_chat_history(
                    user_id=user_id,
                    messages=chat_history,
                    history_type=request.survey_id,
                )
                return SurveyResponse(
                    question=confirmation_question,
                    user_context=user_context,
                    survey_complete=False,
                    intent="SYSTEM_CONFIRM",
                    intent_related_response=None,
                    results_to_save=[],
                    failure_count=request.failure_count,
                )
        else:  # Handle chitchat or other non-journey intents
            rephrased_question = handle_conversational_repair(
                flow_id=flow_id,
                question_identifier=last_step_title,
                previous_question=last_question,
                invalid_input=user_input,
            )
            return SurveyResponse(
                question=rephrased_question,
                user_context=user_context,
                survey_complete=False,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=request.failure_count + 1,
            )

        if user_context == previous_context:
            if request.failure_count >= 1:
                user_context[last_step_title] = "Skipped - System"
            else:
                rephrased_question = handle_conversational_repair(
                    flow_id=flow_id,
                    question_identifier=last_step_title,
                    previous_question=last_question,
                    invalid_input=user_input,
                )
                return SurveyResponse(
                    question=rephrased_question,
                    user_context=previous_context,
                    survey_complete=False,
                    intent="REPAIR",
                    intent_related_response=None,
                    results_to_save=[],
                    failure_count=request.failure_count + 1,
                )

    # --- 4. GET NEXT QUESTION ---
    question_result = await get_anc_survey_question(
        user_id=user_id, user_context=user_context
    )

    question, next_step, survey_complete = "", "", True
    if question_result:
        question = question_result.get("contextualized_question", "")
        survey_complete = question_result.get("is_final_step", False)
        next_step = question_result.get("question_identifier", "")

    if not question and not survey_complete:
        survey_complete = True
        question = "Thank you for completing the survey!"

    if question:
        chat_history.append(
            ChatMessage.from_assistant(text=question, meta={"step_title": next_step})
        )

    await save_chat_history(
        user_id=user_id, messages=chat_history, history_type=request.survey_id
    )
    diff_keys = [
        k for k in user_context if user_context.get(k) != previous_context.get(k)
    ]

    return SurveyResponse(
        question=question or "",
        user_context=user_context,
        survey_complete=survey_complete,
        intent=intent,
        intent_related_response=intent_related_response,
        results_to_save=diff_keys,
        failure_count=0,
    )


@app.post("/v1/catchall", response_model=CatchAllResponse)
async def catchall(request: CatchAllRequest, token: str = Depends(verify_token)):
    intent, intent_related_response = handle_user_message("", request.user_input)

    return CatchAllResponse(
        intent=intent,
        intent_related_response=intent_related_response,
    )
