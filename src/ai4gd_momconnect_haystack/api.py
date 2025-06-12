from os import environ
from typing import Annotated, Any

import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from .tasks import (
    extract_onboarding_data_from_response,
    get_assessment_question,
    get_next_onboarding_question,
    validate_assessment_answer,
)

load_dotenv()


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
    user_input: str
    user_context: dict[str, Any]
    chat_history: list[str] = []


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    chat_history: list[str]


@app.post("/v1/onboarding")
def onboarding(request: OnboardingRequest, token: str = Depends(verify_token)):
    chat_history = request.chat_history
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
    return OnboardingResponse(
        question=question or "", user_context=user_context, chat_history=chat_history
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


@app.post("/v1/assessment")
def assessment(request: AssessmentRequest, token: str = Depends(verify_token)):
    chat_history = request.chat_history
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
    contextualized_question = question["contextualized_question"] or ""
    current_question_number = question["current_question_number"]
    chat_history.append(f"System to User: {contextualized_question}")
    return AssessmentResponse(
        question=contextualized_question,
        next_question=current_question_number,
        chat_history=chat_history,
    )
