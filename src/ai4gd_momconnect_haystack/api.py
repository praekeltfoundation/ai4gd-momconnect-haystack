from os import environ
from pathlib import Path
from typing import Annotated, Any

import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from haystack.dataclasses import ChatMessage
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from .tasks import (
    assessment_map_to_assessment_types,
    assessment_end_flow_map,
    extract_anc_data_from_response,
    extract_onboarding_data_from_response,
    get_anc_survey_question,
    get_assessment_question,
    validate_assessment_end_response,
    get_next_onboarding_question,
    handle_user_message,
    validate_assessment_answer,
)
from .utilities import (
    AssessmentType,
    get_assessment_end_messaging_history,
    get_or_create_chat_history,
    HistoryType,
    load_json_and_validate,
    save_assessment_end_message,
    save_chat_history,
    save_assessment_question,
)

load_dotenv()

DATA_PATH = Path("src/ai4gd_momconnect_haystack/")

SERVICE_PERSONA_PATH = DATA_PATH / "static_content" / "service_persona.json"
SERVICE_PERSONA = load_json_and_validate(SERVICE_PERSONA_PATH, dict)


def setup_sentry():
    if dsn := environ.get("SENTRY_DSN"):
        sentry_sdk.init(dsn=dsn, send_default_pii=True)


setup_sentry()

app = FastAPI()

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


class OnboardingRequest(BaseModel):
    user_id: str
    user_input: str
    user_context: dict[str, Any]


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    intent: str | None
    intent_related_response: str | None


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
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        # Also initialize the chat history with the persona in a system message
        chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA)]
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


class AssessmentRequest(BaseModel):
    user_id: str
    user_input: str
    user_context: dict[str, Any]
    flow_id: AssessmentType
    question_number: int
    previous_question: str


class AssessmentResponse(BaseModel):
    question: str
    next_question: int
    intent: str | None
    intent_related_response: str | None


@app.post("/v1/assessment")
async def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    if request.user_input:
        intent, intent_related_response = handle_user_message(
            request.previous_question, request.user_input
        )
    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""

    if intent == "JOURNEY_RESPONSE" and request.user_input:
        answer = validate_assessment_answer(
            user_response=request.user_input,
            question_number=request.question_number,
            current_flow_id=request.flow_id.value,
        )
        if answer["processed_user_response"]:
            await save_assessment_question(
                user_id=request.user_id,
                assessment_type=request.flow_id,
                question_number=next_question_number,
                user_response=answer["processed_user_response"],
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
        )
    return AssessmentResponse(
        question=contextualized_question,
        next_question=next_question_number,
        intent=intent,
        intent_related_response=intent_related_response,
    )

class AssessmentEndRequest(BaseModel):
    user_id: str
    user_input: str
    user_context: dict[str, Any]
    flow_id: AssessmentType
    score_category: str


class AssessmentEndResponse(BaseModel):
    message: str
    task: str
    intent: str | None
    intent_related_response: str | None


@app.post("/v1/assessment-end")
async def assessment_end(request: AssessmentEndRequest, token: str = Depends(verify_token)):
    messaging_history = await get_assessment_end_messaging_history(
        user_id=request.user_id, assessment_type=request.flow_id
    )
    flow_content = assessment_end_flow_map[request.flow_id.value]
    task = ""

    if request.user_input:
        # Get the previous question from the chat history,
        # which should exist because the endpoint should be called
        # with an empty user_input otherwise.
        previous_message_nr = messaging_history[-1].message_number
        previous_message_data = [item for item in flow_content if item["message_nr"]==previous_message_nr][-1]
        # If previous message number was 1, then content is based on score category
        if previous_message_nr==1:
            if request.score_category=='high':
                previous_message = previous_message_data["high-score-content"]["content"]
                previous_message_valid_responses = previous_message_data["high-score-content"]["valid_responses"]
            elif request.score_category=='medium':
                previous_message = previous_message_data["medium-score-content"]["content"]
                previous_message_valid_responses = previous_message_data["medium-score-content"]["valid_responses"]
            elif request.score_category=='low':
                previous_message = previous_message_data["low-score-content"]["content"]
                previous_message_valid_responses = previous_message_data["low-score-content"]["valid_responses"]
            else:
                previous_message = previous_message_data["skipped-many-content"]["content"]
                previous_message_valid_responses = previous_message_data["skipped-many-content"]["valid_responses"]
        # For previous message numbers 2/3, content is the same across score categories.
        else:
            previous_message = previous_message_data["content"]
            previous_message_valid_responses = previous_message_data["valid_responses"]
        intent, intent_related_response = handle_user_message(
            previous_message, request.user_input
        )
        next_message_nr = previous_message_nr + 1
    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        next_message_nr = 1

    # If the user responded to the previous question, we process their response and determine what needs to happen next
    if intent == "JOURNEY_RESPONSE" and request.user_input:
        # If the user responded to a question that demands a response
        if (request.flow_id.value=="dma-pre-assessment" and next_message_nr in [2,3]) or (request.flow_id.value=="behaviour-pre-assessment" and next_message_nr == 2) or (request.flow_id.value=="knowledge-pre-assessment" and next_message_nr == 2) or (request.flow_id.value=="attitude-pre-assessment" and next_message_nr == 2):
            result = validate_assessment_end_response(
                previous_message=previous_message,
                previous_message_number=next_message_nr-1,
                previous_message_valid_responses=previous_message_valid_responses,
                user_response=request.user_input,
            )
            if result["next_message_number"]==next_message_nr-1:
                # validation failed, send previous message again
                next_message_data = previous_message_data
            else:
                processed_user_response = result["processed_user_response"]
                # If the user response was valid, save it to the existing AssessmentEndMessagingHistory record
                await save_assessment_end_message(request.user_id, request.flow_id, next_message_nr-1, processed_user_response)
                # validation succeeded, so determine next message to send
                next_message_data = [item for item in flow_content if item["message_nr"]==next_message_nr][-1]
                
                # Now determine the task that's associated with the response
                if request.flow_id.value == "dma-pre-assessment":
                    if previous_message_nr == 1 and request.score_category == "skipped-many":
                        if processed_user_response == "Yes":
                            task = "REMIND_ME_LATER"
                    elif previous_message_nr == 2:
                        task = "STORE_FEEDBACK"
                elif request.flow_id.value == "behaviour-pre-assessment":
                    if previous_message_nr == 1 and processed_user_response == "Remind me tomorrow":
                        task = "REMIND_ME_LATER"
                elif request.flow_id.value == "knowledge-pre-assessment":
                    if previous_message_nr == 1 and processed_user_response == "Remind me tomorrow":
                        task = "REMIND_ME_LATER"
                elif request.flow_id.value == "attitude-pre-assessment":
                    if previous_message_nr == 1:
                        if request.score_category == "skipped-many":
                            if processed_user_response == "Yes":
                                task = "REMIND_ME_LATER"
                        else:
                            task = "STORE_FEEDBACK"
            next_message = next_message_data["content"]
        # Else the user responded to the last question, which doesn't require a response
        else:
            # Here we return an empty message because the user responded by the journey ended
            next_message = ""
            
    elif intent=="JOURNEY_RESPONSE":
        # This triggers if the journey is initiating (i.e. the next message to be sent is the first)
        next_message_data = [item for item in flow_content if item["message_nr"]==next_message_nr][-1]
        if request.score_category=='high':
            next_message = next_message_data["high-score-content"]["content"]
        elif request.score_category=='medium':
            next_message = next_message_data["medium-score-content"]["content"]
        elif request.score_category=='low':
            next_message = next_message_data["low-score-content"]["content"]
        else:
            next_message = next_message_data["skipped-many-content"]["content"]
    else:
        # It doesn't seem like the user is responding to the journey, and neither is it their first question, so we ask them the previous question again.
        next_message_data = [item for item in flow_content if item["message_nr"]==previous_message_nr][-1]
        next_message = next_message_data["content"]
    
    
    # If a new question is being sent, save it in a new AssessmentEndMessagingHistory record
    if next_message and next_message_nr==previous_message_nr+1:
        await save_assessment_end_message(request.user_id, request.flow_id, next_message_nr, "")
    return AssessmentEndResponse(
        message=next_message,
        task=task,
        intent=intent,
        intent_related_response=intent_related_response,
    )


class SurveyRequest(BaseModel):
    user_id: str
    survey_id: HistoryType
    user_input: str
    user_context: dict[str, Any]


class SurveyResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    survey_complete: bool
    intent: str | None
    intent_related_response: str | None


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

    last_question = chat_history[-1] if chat_history else ""
    if request.user_input:
        chat_history.append(ChatMessage.from_user(text=user_input))
        intent, intent_related_response = handle_user_message(last_question, user_input)
    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA)]
    if intent == "JOURNEY_RESPONSE" and request.user_input:
        # There's only one survey type, so we can assume anc until we add more
        # First, extract data from the user's last response to update the context
        user_context = extract_anc_data_from_response(
            user_response=request.user_input,
            user_context=request.user_context,
            chat_history=chat_history,
        )
    else:
        user_context = request.user_context
    # Then, get the next logical question based on the updated context
    question_result = get_anc_survey_question(
        user_context=user_context, chat_history=chat_history
    )
    question = ""
    survey_complete = True  # Default to True if no further question is found
    if question_result:
        question = question_result.get("contextualized_question", "")
        survey_complete = question_result.get("is_final_step", False)
    # Add the new question or a completion message to the history
    if (not question) and survey_complete:
        completion_message = "Thank you for completing the survey!"
        question = completion_message
    chat_history.append(ChatMessage.from_assistant(text=question))
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
