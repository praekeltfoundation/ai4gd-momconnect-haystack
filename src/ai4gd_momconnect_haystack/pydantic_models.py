# models.py

from typing import Any
from pydantic import BaseModel, Field, model_validator


# NEW: A model to represent a single response and its score
class ResponseScore(BaseModel):
    response: str
    score: int


class Question(BaseModel):
    """
    Defines a flexible structure for any question from the doc store.
    It can handle scorable (assessment) and non-scorable (onboarding) questions.
    """

    question_number: int
    question_name: str | None = None
    content: str | None = None
    pre_content: str | None = Field(None, alias="pre-content")
    post_content: str | None = Field(None, alias="post-content")

    # UPDATED: Replaced the old `valid_responses` with the new structure
    valid_responses_and_scores: list[ResponseScore] | list[str] | None = Field(
        None, alias="valid_responses_and_scores"
    )

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
    follow_up_utterance: str | None = None # To store the second user message
    user_response: str | None = Field(None, alias="llm_extracted_user_response")
    llm_initial_predicted_intent: str | None = Field(None, alias="llm_initial_predicted_intent")
    llm_final_predicted_intent: str | None = Field(None, alias="llm_final_predicted_intent")

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
