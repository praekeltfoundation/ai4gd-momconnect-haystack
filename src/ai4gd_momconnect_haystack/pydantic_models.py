# models.py

from datetime import datetime, timedelta
from typing import Any, TypeVar, TypedDict

from pydantic import BaseModel, Field, model_validator

from .enums import (
    AssessmentType,
    HistoryType,
)

import uuid
from dataclasses import dataclass, field

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
    follow_up_utterance: str | None = None  # To store the second user message
    user_response: str | dict | None = Field(None, alias="llm_extracted_user_response")
    llm_initial_predicted_intent: str | None = Field(
        None, alias="llm_initial_predicted_intent"
    )
    llm_final_predicted_intent: str | None = Field(
        None, alias="llm_final_predicted_intent"
    )

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
    failure_count: int = 0


class ReengagementInfo(BaseModel):
    type: str
    trigger_at_utc: datetime
    flow_id: str
    reminder_type: int


class OnboardingResponse(BaseModel):
    question: str
    user_context: dict[str, Any]
    intent: str | None
    intent_related_response: str | None
    results_to_save: list[str]
    failure_count: int
    reengagement_info: ReengagementInfo | None = None


class AssessmentRequest(BaseModel):
    user_id: str
    user_input: str
    user_context: dict[str, Any]
    flow_id: AssessmentType
    question_number: int
    previous_question: str
    failure_count: int = 0


class AssessmentResponse(BaseModel):
    question: str
    next_question: int | None
    intent: str | None
    intent_related_response: str | None
    processed_answer: str | None
    failure_count: int = 0
    reengagement_info: ReengagementInfo | None = None


class AssessmentEndRequest(BaseModel):
    user_id: str
    user_input: str
    flow_id: AssessmentType


class AssessmentEndResponse(BaseModel):
    message: str
    task: str | None
    intent: str | None
    intent_related_response: str | None
    reengagement_info: ReengagementInfo | None = None


class SurveyRequest(BaseModel):
    user_id: str
    survey_id: str
    user_input: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)
    is_re_engagement_ping: bool = False
    turn_id: str | None = None
    trace_id: str | None = None


class SurveyResponse(BaseModel):
    question: str
    question_identifier: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)
    survey_complete: bool = False
    intent: str | None = None
    intent_related_response: str | None = None
    results_to_save: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    reengagement_info: ReengagementInfo | None = None


class IntroductionMessage(BaseModel):
    id: str
    content: str


class CatchAllRequest(BaseModel):
    user_id: str
    user_input: str


class CatchAllResponse(BaseModel):
    intent: str | None
    intent_related_response: str | None


class ResumeRequest(BaseModel):
    user_id: str


class ResumeResponse(BaseModel):
    resume_flow_id: str


class ReminderConfig(TypedDict):
    delay: timedelta
    acknowledgement_message: str
    resume_message: str | None


# --- NEW: Internal Data Transfer Objects ---


class ExtractedData(BaseModel):
    is_success: bool
    updated_context: dict[str, Any] = Field(default_factory=dict)
    action_dict: dict[str, Any] | None = None
    reason: str | None = None  # FIX: Added missing 'reason' field


class IntentResult(BaseModel):
    intent: str
    intent_related_response: str | None = None
    reason: str | None = None  # FIX: Added missing 'reason' field


class UserJourneyState(BaseModel):
    """Pydantic version for use within the application logic."""

    user_id: str
    flow_id: str
    expected_step_id: str | None = None
    last_question: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)
    repair_strikes: dict[str, int] = Field(default_factory=dict)
    version: int = 0


@dataclass
class SurveyTurnContext:
    request: SurveyRequest
    journey_state: UserJourneyState | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    previous_context: dict[str, Any] = field(default_factory=dict)
    current_context: dict[str, Any] = field(default_factory=dict)
    last_assistant_message: dict[str, Any] | None = None
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class LegacySurveyRequest(BaseModel): # Renamed to avoid conflict
    user_id: str
    survey_id: HistoryType
    user_input: str
    user_context: dict[str, Any]
    failure_count: int = 0

class LegacySurveyResponse(BaseModel): # Renamed to avoid conflict
    question: str | None
    question_identifier: str | None = None
    user_context: dict[str, Any]
    survey_complete: bool
    intent: str | None
    intent_related_response: str | None
    results_to_save: list[str]
    failure_count: int = 0
    reengagement_info: ReengagementInfo | None = None
