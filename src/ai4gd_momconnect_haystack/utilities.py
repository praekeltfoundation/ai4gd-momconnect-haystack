from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

from haystack.dataclasses import ChatMessage
from pydantic import BaseModel, ValidationError
from sqlalchemy import delete
from sqlalchemy.future import select

from .database import AsyncSessionLocal
from .sqlalchemy_models import ChatHistory, PreAssessmentQuestionHistory


logger = logging.getLogger(__name__)


class AssessmentType(str, Enum):
    dma_pre_assessment = "dma-pre-assessment"
    dma_post_assessment = "dma-post-assessment"
    knowledge_pre_assessment = "knowledge-pre-assessment"
    knowledge_post_assessment = "knowledge-post-assessment"
    attitude_pre_assessment = "attitude-pre-assessment"
    attitude_post_assessment = "attitude-post-assessment"
    behaviour_pre_assessment = "behaviour-pre-assessment"
    behaviour_post_assessment = "behaviour-post-assessment"


class HistoryType(str, Enum):
    anc = "anc"
    onboarding = "onboarding"


def read_json(filepath: Path) -> dict:
    """Reads JSON data from a file."""
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading JSON from %s: %s", filepath, e)
        raise


def generate_scenario_id(flow_type: str, username: str) -> str:
    """Generates a unique scenario ID."""
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return f"{flow_type}_{username}_{timestamp}"


def load_json_and_validate(
    file_path: Path, model: type[BaseModel] | type[dict]
) -> Any | None:
    """
    Loads a JSON file and validates its content against a Pydantic model or as a dict.
    This is the primary gateway for safely loading any external JSON data.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Guard Clause: If the model is just 'dict', we are loading a raw
        # doc_store. Return it directly without Pydantic validation.
        # The validation for its contents is handled later in tasks.py.
        if model is dict:
            return raw_data

        # If the model is a Pydantic model, proceed with validation.
        if issubclass(model, BaseModel):
            if isinstance(raw_data, list):
                return [model.model_validate(item) for item in raw_data]
            return model.model_validate(raw_data)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except ValidationError as e:
        logging.error(f"Data validation error in {file_path}:\n{e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred with {file_path}: {e}")
    return None


def save_json_file(data: list[dict[str, Any]], file_path: Path) -> None:
    """Saves the final processed data to a JSON file."""
    try:
        # Ensure the output directory exists before writing.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Successfully saved final augmented output to {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing to {file_path}: {e}")


def chat_messages_to_json(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Converts a list of ChatMessage objects to a JSON-serializable list of dicts."""
    return [
        {
            "role": msg.role.value,
            "text": msg.text,
        }
        for msg in messages
    ]


async def get_or_create_chat_history(
    user_id: str, history_type: HistoryType
) -> list[ChatMessage]:
    """
    Retrieves an Onboarding chat history for a given user_id.
    If it doesn't exist, an empty history is returned.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatHistory).filter(ChatHistory.user_id == user_id)
        )
        db_history = result.scalar_one_or_none()
        chat_history = []

        if db_history:
            if history_type == HistoryType.onboarding:
                db_chat_history = db_history.onboarding_history
            elif history_type == HistoryType.anc:
                db_chat_history = db_history.anc_survey_history

            for cm in db_chat_history:
                if cm["role"] == "user":
                    chat_history.append(ChatMessage.from_user(text=cm["text"]))
                elif cm["role"] == "assistant":
                    chat_history.append(ChatMessage.from_assistant(text=cm["text"]))
                else:
                    chat_history.append(ChatMessage.from_system(text=cm["text"]))
        return chat_history


async def save_chat_history(
    user_id: str, messages: list[ChatMessage], history_type: HistoryType
) -> None:
    """Saves or updates the chat history for a given user_id."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(ChatHistory).filter(ChatHistory.user_id == user_id)
            )
            db_history = result.scalar_one_or_none()

            history_json = chat_messages_to_json(messages)

            if db_history:
                if history_type == "onboarding":
                    db_history.onboarding_history = history_json
                elif history_type == "anc":
                    db_history.anc_survey_history = history_json
                else:
                    logger.error(f"Unknown chat history type to update: {history_type}")
                    return
            else:
                if history_type == "onboarding":
                    db_history = ChatHistory(
                        user_id=user_id,
                        onboarding_history=history_json,
                        anc_survey_history=[],
                    )
                elif history_type == "anc":
                    db_history = ChatHistory(
                        user_id=user_id,
                        onboarding_history=[],
                        anc_survey_history=history_json,
                    )
                else:
                    logger.error(f"Unknown chat history type to create: {history_type}")
                    return
                session.add(db_history)

            await session.commit()


async def get_pre_assessment_history(
    user_id: str, assessment_type: AssessmentType
) -> list[PreAssessmentQuestionHistory]:
    """
    Retrieves pre-assessment question history for a given user and assessment type.
    If no history exists, an empty list is returned.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PreAssessmentQuestionHistory).filter(
                PreAssessmentQuestionHistory.user_id == user_id,
                PreAssessmentQuestionHistory.assessment_id == assessment_type.value,
            )
        )
        history = result.scalars().all()
        return list(history)


async def save_pre_assessment_question(
    user_id: str,
    assessment_type: AssessmentType,
    question_number: int,
    question: str,
) -> None:
    """
    Saves the pre-assessment question history for a given user, assessment type and question number.
    This will overwrite any existing history for the same user, assessment type and question number.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # First, delete any existing history for this user, assessment type and question number.
            await session.execute(
                delete(PreAssessmentQuestionHistory).where(
                    PreAssessmentQuestionHistory.user_id == user_id,
                    PreAssessmentQuestionHistory.assessment_id == assessment_type.value,
                    PreAssessmentQuestionHistory.question_number == question_number,
                )
            )
            new_historic_record = PreAssessmentQuestionHistory(
                user_id=user_id,
                assessment_id=assessment_type.value,
                question_number=question_number,
                question=question,
            )
            session.add(new_historic_record)
            await session.commit()
