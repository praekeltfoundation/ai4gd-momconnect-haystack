from collections import defaultdict
from os import environ
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .tasks import (
    extract_onboarding_data_from_response,
    get_assessment_question,
    get_next_onboarding_question,
    validate_assessment_answer,
)

load_dotenv()
API_TOKEN = environ["API_TOKEN"]

app = FastAPI()

CHAT_HISTORY = defaultdict(list)


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

    if credential != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
        )

    return credential


class OnboardingRequest(BaseModel):
    whatsapp_id: str
    user_input: str
    user_context: dict[str, Any]


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]


@app.post("/v1/onboarding")
def onboarding(request: OnboardingRequest, token: str = Depends(verify_token)):
    chat_history = CHAT_HISTORY[request.whatsapp_id]
    chat_history.append(f"User to System: {request.user_input}")
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
    return OnboardingResponse(question=question, user_context=user_context)


class AssessmentRequest(BaseModel):
    whatsapp_id: str
    user_input: str
    user_context: dict[str, Any]
    flow_id: str
    question_number: int


class AssessmentResponse(BaseModel):
    question: str
    next_question: int


@app.post("/v1/assessment")
def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    chat_history = CHAT_HISTORY[request.whatsapp_id]
    chat_history.append(f"User to System: {request.user_input}")
    # TODO: Can we run these in parallel for latency?
    validate_assessment_answer(
        user_response=request.user_input,
        current_question_number=request.question_number,
    )
    question = get_assessment_question(
        flow_id=request.flow_id,
        question_number=request.question_number,
        current_assessment_step=request.question_number,
        user_context=request.user_context,
    )
    contextualized_question = question["contextualized_question"]
    current_question_number = question["current_question_number"]
    chat_history.append(f"System to User: {contextualized_question}")
    return AssessmentResponse(
        question=contextualized_question, next_question=current_question_number
    )
