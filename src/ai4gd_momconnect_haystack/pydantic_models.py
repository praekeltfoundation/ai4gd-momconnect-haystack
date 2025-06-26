# models.py

from typing import Any, TypeVar
from pydantic import BaseModel, Field, model_validator

from .enums import (
    AssessmentType,
    HistoryType,
)


# A generic model type
T = TypeVar("T", bound=BaseModel)


class ResponseScore(BaseModel):
    response: str
    score: int


class AssessmentResult(BaseModel):
    score: float
    category: str
    crossed_skip_threshold: bool


class AssessmentQuestion(BaseModel):
    """
    Defines a flexible structure for any question from the doc store.
    It can handle scorable (assessment) and non-scorable (onboarding) questions.
    """

    question_number: int
    question_name: str | None = None
    content: str | None = None
    valid_responses_and_scores: list[ResponseScore] | None = Field(
        None, alias="valid_responses_and_scores"
    )
    content_type: str | None = None


class OnboardingQuestion(BaseModel):
    """
    Defines a flexible structure for any question from the doc store.
    It can handle scorable (assessment) and non-scorable (onboarding) questions.
    """

    question_number: int
    content: str | None = None
    valid_responses: list[str] | None = Field(None, alias="valid_responses")
    content_type: str | None = None
    collects: str | None = None
    reason: str | None = None


class Turn(BaseModel):
    """
    Defines a single turn in a conversation.
    The validator ensures it has at least one valid identifier.
    """

    question_name: str | None = None
    question_number: int | None = None
    llm_utterance: str | None = None
    user_utterance: str | None = None
    user_response: str | None = Field(None, alias="llm_extracted_user_response")

    @model_validator(mode="before")
    @classmethod
    def check_at_least_one_identifier_exists(cls, data: Any) -> Any:
        """Ensures each turn has 'question_name' or 'question_number'."""
        if isinstance(data, dict):
            if "question_name" not in data and "question_number" not in data:
                raise ValueError(
                    "Turn must have either 'question_name' or 'question_number'"
                )
        return data


class AssessmentRun(BaseModel):
    """Defines a full assessment run from the simulation output."""

    scenario_id: str
    flow_type: str
    turns: list[Turn]


class FAQ(BaseModel):
    title: str
    content: str


class ANCSurveyQuestion(BaseModel):
    title: str
    content: str
    content_type: str
    valid_responses: list[str] | None = Field(None, alias="valid_responses")


class AssessmentEndContentItem(BaseModel):
    content: str
    valid_responses: list[str] | None = Field(None, alias="valid_responses")


class AssessmentEndScoreBasedMessage(BaseModel):
    message_nr: int
    high_score_content: AssessmentEndContentItem = Field(
        ..., alias="high-score-content"
    )
    medium_score_content: AssessmentEndContentItem = Field(
        ..., alias="medium-score-content"
    )
    low_score_content: AssessmentEndContentItem = Field(..., alias="low-score-content")
    skipped_many_content: AssessmentEndContentItem = Field(
        ..., alias="skipped-many-content"
    )


class AssessmentEndSimpleMessage(BaseModel):
    message_nr: int
    content: str
    valid_responses: list[str] | None = Field(None, alias="valid_responses")


AssessmentEndItem = AssessmentEndScoreBasedMessage | AssessmentEndSimpleMessage


### API Request and Response models ###
class OnboardingRequest(BaseModel):
    user_id: str
    user_input: str
    user_context: dict[str, Any]


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    intent: str | None
    intent_related_response: str | None


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


class AssessmentEndRequest(BaseModel):
    user_id: str
    user_input: str
    flow_id: AssessmentType


class AssessmentEndResponse(BaseModel):
    message: str
    task: str
    intent: str | None
    intent_related_response: str | None


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
