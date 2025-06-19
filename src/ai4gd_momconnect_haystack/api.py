from enum import Enum
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
    extract_anc_data_from_response,
    extract_onboarding_data_from_response,
    get_anc_survey_question,
    get_assessment_question,
    get_next_onboarding_question,
    handle_user_message,
    validate_assessment_answer,
)
from .utilities import (
    ChatTypes,
    get_or_create_chat_history,
    load_json_and_validate,
    save_chat_history,
    SurveyTypes,
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
        user_id=user_id, history_type=ChatTypes.onboarding
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
    if intent == "CHITCHAT" and intent_related_response:
        chat_history.append(ChatMessage.from_assistant(text=intent_related_response))
    question = get_next_onboarding_question(
        user_context=user_context, chat_history=chat_history
    )
    if question:
        chat_history.append(ChatMessage.from_assistant(text=question))
    await save_chat_history(
        user_id=request.user_id,
        messages=chat_history,
        history_type=ChatTypes.onboarding,
    )
    return OnboardingResponse(
        question=question or "",
        user_context=user_context,
        intent=intent,
        intent_related_response=intent_related_response,
    )


class AssessmentType(str, Enum):
    dma_assessment = "dma-assessment"
    knowledge_assessment = "knowledge-assessment"
    attitude_assessment = "attitude-assessment"
    behaviour_pre_assessment = "behaviour-pre-assessment"
    behaviour_post_assessment = "behaviour-post-assessment"


class AssessmentRequest(BaseModel):
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
def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    if request.user_input:
        intent, intent_related_response = handle_user_message(
            request.previous_question, request.user_input
        )
    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""

    if intent == "JOURNEY_RESPONSE" and request.user_input:
        answer = validate_assessment_answer(
            user_response=request.user_input,
            current_question_number=request.question_number - 1,
            current_flow_id=request.flow_id,
        )
        current_question_number = answer["current_assessment_step"]
    else:
        current_question_number = request.question_number

    question = get_assessment_question(
        flow_id=request.flow_id,
        current_assessment_step=current_question_number,
        user_context=request.user_context,
    )
    contextualized_question = question["contextualized_question"] or ""
    current_question_number = question["current_question_number"]
    return AssessmentResponse(
        question=contextualized_question,
        next_question=current_question_number,
        intent=intent,
        intent_related_response=intent_related_response,
    )


class SurveyRequest(BaseModel):
    user_id: str
    survey_id: SurveyTypes
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
        await save_chat_history(
            user_id=request.user_id,
            messages=chat_history,
            history_type=request.survey_id,
        )
        intent, intent_related_response = handle_user_message(last_question, user_input)
    else:
        intent, intent_related_response = "JOURNEY_RESPONSE", ""
        if not chat_history:
            chat_history = [ChatMessage.from_system(text=SERVICE_PERSONA)]
            await save_chat_history(
                user_id=user_id, messages=chat_history, history_type=request.survey_id
            )
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
