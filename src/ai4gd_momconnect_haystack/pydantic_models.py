# models.py

from typing import Any
from pydantic import BaseModel, Field, model_validator


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
    valid_responses: dict[str, int] | list[str]
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
    user_response: str = Field(alias="llm_extracted_user_response")

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
