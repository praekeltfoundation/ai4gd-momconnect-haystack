from enum import Enum
from os import environ
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
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

load_dotenv()

app = FastAPI()


@app.get("/health")
def health():
    # TODO: Proper health check
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
    user_input: str
    user_context: dict[str, Any]
    chat_history: list[str] = []


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    chat_history: list[str]
    intent: str | None
    intent_related_response: str | None


@app.post("/v1/onboarding")
def onboarding(request: OnboardingRequest, token: str = Depends(verify_token)):
    chat_history = request.chat_history
    last_question = chat_history[-1] if chat_history else ""
    intent, intent_related_response = handle_user_message(
        last_question, request.user_input
    )
    chat_history.append(f"User to System: {request.user_input}")
    if intent == "JOURNEY_RESPONSE":
        # TODO: Can we run these in parallel for latency?
        user_context = extract_onboarding_data_from_response(
            user_response=request.user_input,
            user_context=request.user_context,
            chat_history=chat_history,
        )
        question = get_next_onboarding_question(
            user_context=user_context, chat_history=chat_history
        )
        chat_history.append(f"System to User: {question}")
    else:
        user_context = request.user_context
        question = ""
    return OnboardingResponse(
        question=question or "",
        user_context=user_context,
        chat_history=chat_history,
        intent=intent,
        intent_related_response=intent_related_response,
    )


class AssessmentRequest(BaseModel):
    user_input: str
    user_context: dict[str, Any]
    flow_id: str
    question_number: int
    chat_history: list[str] = []


class AssessmentResponse(BaseModel):
    question: str
    next_question: int
    chat_history: list[str]
    intent: str | None
    intent_related_response: str | None


@app.post("/v1/assessment")
def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    chat_history = request.chat_history
    last_question = chat_history[-1] if chat_history else ""
    intent, intent_related_response = handle_user_message(
        last_question, request.user_input
    )
    chat_history.append(f"User to System: {request.user_input}")
    if intent == "JOURNEY_RESPONSE":
        # TODO: Can we run these in parallel for latency?
        validate_assessment_answer(
            user_response=request.user_input,
            current_question_number=request.question_number,
            current_flow_id=request.flow_id,
        )
        question = get_assessment_question(
            flow_id=request.flow_id,
            current_assessment_step=request.question_number,
            user_context=request.user_context,
        )
        contextualized_question = question["contextualized_question"] or ""
        current_question_number = question["current_question_number"]
        chat_history.append(f"System to User: {contextualized_question}")
    else:
        current_question_number = request.question_number
        contextualized_question = ""
    return AssessmentResponse(
        question=contextualized_question,
        next_question=current_question_number,
        chat_history=chat_history,
        intent=intent,
        intent_related_response=intent_related_response,
    )


class SurveyTypes(str, Enum):
    anc = "anc"


class SurveyRequest(BaseModel):
    survey_id: SurveyTypes
    user_input: str
    user_context: dict[str, Any]
    chat_history: list[str] = []


class SurveyResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    chat_history: list[str]
    survey_complete: bool
    intent: str | None
    intent_related_response: str | None


@app.post("/v1/survey", response_model=SurveyResponse)
def anc_survey(request: SurveyRequest, token: str = Depends(verify_token)):
    """
    Handles the conversation flow for the ANC (Antenatal Care) survey.
    It extracts data from the user's response and determines the next question.
    """
    chat_history = request.chat_history
    last_question = chat_history[-1] if chat_history else ""
    intent, intent_related_response = handle_user_message(
        last_question, request.user_input
    )
    chat_history.append(f"User to System: {request.user_input}")
    if intent == "JOURNEY_RESPONSE":
        # There's only one survey type, so we can assume anc until we add more
        # First, extract data from the user's last response to update the context
        user_context = extract_anc_data_from_response(
            user_response=request.user_input,
            user_context=request.user_context,
            chat_history=chat_history,
        )
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
        if question:
            chat_history.append(f"System to User: {question}")
        elif survey_complete:
            completion_message = "Thank you for completing the survey!"
            chat_history.append(f"System to User: {completion_message}")
            question = completion_message
    else:
        user_context = request.user_context
        question = ""
        survey_complete = False
    return SurveyResponse(
        question=question,
        user_context=user_context,
        chat_history=chat_history,
        survey_complete=survey_complete,
        intent=intent,
        intent_related_response=intent_related_response,
    )
