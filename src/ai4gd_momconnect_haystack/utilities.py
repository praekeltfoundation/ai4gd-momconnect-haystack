from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from haystack.dataclasses import ChatMessage
from pydantic import BaseModel, ValidationError
from sqlalchemy import delete, update
from sqlalchemy.future import select

from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndContentItem,
    AssessmentEndScoreBasedMessage,
    AssessmentEndSimpleMessage,
    AssessmentResult,
    AssessmentRun,
    Turn,
)
from ai4gd_momconnect_haystack.tasks import (
    assessment_flow_map,
    assessment_map_to_their_pre,
    _calculate_assessment_score_range,
    dma_pre_flow_id,
    dma_post_flow_id,
    kab_a_pre_flow_id,
    kab_a_post_flow_id,
    kab_b_post_flow_id,
    kab_k_pre_flow_id,
    kab_k_post_flow_id,
    load_and_validate_assessment_questions,
    _score_single_turn,
)

from .database import AsyncSessionLocal
from .sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    AssessmentResultHistory,
    ChatHistory,
    AssessmentHistory,
)


logger = logging.getLogger(__name__)


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


def score_assessment(
    assessment_run: AssessmentRun,
    assessment_id: AssessmentType,
) -> AssessmentResult | None:
    assessment_questions = load_and_validate_assessment_questions(
        assessment_map_to_their_pre[assessment_id.value]
    )
    if not assessment_questions:
        logger.error(
            "Function 'load_and_validate_assessment_questions' called incorrectly"
        )
        return None
    question_lookup = {q.question_number: q for q in assessment_questions}

    # ---Score Calculation Logic ---
    min_possible_score, max_possible_score = _calculate_assessment_score_range(
        assessment_questions
    )
    results = [
        _score_single_turn(turn, question_lookup)
        for turn in assessment_run.turns
        if turn.question_number is not None
    ]
    user_total_score = sum(r["score"] for r in results if r.get("score") is not None)
    score_percentage = (
        (user_total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
    )
    skip_count = sum(turn.user_response == "Skip" for turn in assessment_run.turns)
    category = ""
    crossed_skip_threshold = False
    if assessment_id.value in [dma_pre_flow_id, dma_post_flow_id]:
        if score_percentage > 67:
            category = "high"
        elif score_percentage > 33:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    elif assessment_id.value in [kab_k_pre_flow_id, kab_k_post_flow_id]:
        if score_percentage > 83:
            category = "high"
        elif score_percentage > 50:
            category = "medium"
        else:
            category = "low"
        if skip_count > 3:
            crossed_skip_threshold = True
    elif assessment_id.value in [kab_a_pre_flow_id, kab_a_post_flow_id]:
        if score_percentage >= 80:
            category = "high"
        elif score_percentage >= 60:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    else:
        if score_percentage >= 100:
            category = "high"
        elif score_percentage >= 75:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    return AssessmentResult.model_validate(
        {
            "score": score_percentage,
            "category": category,
            "crossed_skip_threshold": crossed_skip_threshold,
        }
    )


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


async def save_assessment_question(
    user_id: str,
    assessment_type: AssessmentType,
    question_number: int,
    question: str | None,
    user_response: str | None,
    score: int | None,
) -> None:
    """
    Saves or updates a historic assessment question for a given user, assessment type and question number.
    """
    print(
        f"Saving user response '{user_response}' with score '{score}' to question number {question_number}"
    )
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(AssessmentHistory).where(
                    AssessmentHistory.user_id == user_id,
                    AssessmentHistory.assessment_id == assessment_type.value,
                    AssessmentHistory.question_number == question_number,
                )
            )
            historic_record = result.scalar_one_or_none()

            if not historic_record:
                historic_record = AssessmentHistory(
                    user_id=user_id,
                    assessment_id=assessment_type.value,
                    question_number=question_number,
                )
                session.add(historic_record)

            if question is not None:
                historic_record.question = question
            if user_response is not None:
                historic_record.user_response = user_response
            if score is not None:
                historic_record.score = score


async def calculate_and_store_assessment_result(
    user_id: str, assessment_type: AssessmentType
) -> None:
    """
    Checks if an assessment is complete and, if so, calculates and stores the final result.
    """
    # Check if all scores are available for the given assessment
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(AssessmentHistory)
                .where(
                    AssessmentHistory.user_id == user_id,
                    AssessmentHistory.assessment_id == assessment_type.value,
                    AssessmentHistory.score.is_not(None),
                )
                .order_by(AssessmentHistory.question_number.asc())
            )
            historic_records = result.scalars().all()
            print(f"Number of historic records with scores: {len(historic_records)}")

            # If the number of scored records matches the expected length, proceed
            if not matches_assessment_question_length(
                len(historic_records), assessment_type
            ):
                return

            # Calculate and store an AssessmentResultHistory.
            turns = [
                Turn.model_validate(
                    {
                        "question_number": rec.question_number,
                        "user_response": rec.user_response,
                    }
                )
                for rec in historic_records
            ]
            assessment_run = AssessmentRun.model_validate(
                {
                    "scenario_id": user_id,
                    "flow_type": assessment_type.value,
                    "turns": turns,
                }
            )
            assessment_result = score_assessment(assessment_run, assessment_type)

            if not assessment_result:
                return

            # Find or create the result history record
            result_history_stmt = select(AssessmentResultHistory).where(
                AssessmentResultHistory.user_id == user_id,
                AssessmentResultHistory.assessment_id == assessment_type.value,
            )
            result_history_record = (
                await session.execute(result_history_stmt)
            ).scalar_one_or_none()

            if not result_history_record:
                result_history_record = AssessmentResultHistory(
                    user_id=user_id,
                    assessment_id=assessment_type.value,
                )
                session.add(result_history_record)

            # Update the record with the new results
            result_history_record.category = assessment_result.category
            result_history_record.score = assessment_result.score
            result_history_record.crossed_skip_threshold = (
                assessment_result.crossed_skip_threshold
            )


async def get_assessment_result(
    user_id: str, assessment_type: AssessmentType
) -> AssessmentResult | None:
    print(
        f"Trying to get assessment results for {user_id} for assessment {assessment_type.value}"
    )
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                select(AssessmentResultHistory).where(
                    AssessmentResultHistory.user_id == user_id,
                    AssessmentResultHistory.assessment_id == assessment_type.value,
                )
            )
            historic_record = result.scalar_one_or_none()
            print(historic_record)
            if historic_record:
                return AssessmentResult.model_validate(
                    {
                        "score": historic_record.crossed_skip_threshold,
                        "category": historic_record.category,
                        "crossed_skip_threshold": historic_record.crossed_skip_threshold,
                    }
                )
    return None


async def get_assessment_end_messaging_history(
    user_id: str, assessment_type: AssessmentType
) -> list[AssessmentEndMessagingHistory]:
    """
    Retrieves assessment end messaging history for a given user and assessment type.
    If no history exists, an empty list is returned.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AssessmentEndMessagingHistory)
            .filter(
                AssessmentEndMessagingHistory.user_id == user_id,
                AssessmentEndMessagingHistory.assessment_id == assessment_type.value,
            )
            .order_by(AssessmentEndMessagingHistory.message_number.asc())
        )
        history = result.scalars().all()
        return list(history)


async def save_assessment_end_message(
    user_id: str,
    assessment_type: AssessmentType,
    message_number: int,
    user_response: str,
) -> None:
    """
    Saves the end-of-assessment messaging history for a given user, assessment type and question number.
    This will overwrite any existing history for the same user, assessment type and question number.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # If a user_response is present, the question should already exist in the database and we can just update it.
            if user_response:
                await session.execute(
                    update(AssessmentEndMessagingHistory)
                    .where(
                        AssessmentEndMessagingHistory.user_id == user_id,
                        AssessmentEndMessagingHistory.assessment_id
                        == assessment_type.value,
                        AssessmentEndMessagingHistory.message_number == message_number,
                    )
                    .values(user_response=user_response)
                )
            else:
                # First, delete any existing history in case there is one
                await session.execute(
                    delete(AssessmentEndMessagingHistory).where(
                        AssessmentEndMessagingHistory.user_id == user_id,
                        AssessmentEndMessagingHistory.assessment_id
                        == assessment_type.value,
                        AssessmentEndMessagingHistory.message_number == message_number,
                    )
                )
                # Then write a new record
                new_historic_record = AssessmentEndMessagingHistory(
                    user_id=user_id,
                    assessment_id=assessment_type.value,
                    message_number=message_number,
                    user_response=None,
                )
                session.add(new_historic_record)
            await session.commit()


def get_content_from_message_data(
    message_data: AssessmentEndScoreBasedMessage | AssessmentEndSimpleMessage,
    score_category: str,
) -> tuple[str, list[str] | None]:
    """
    Extracts the content and valid responses from a message object based on the score category.
    """
    if isinstance(message_data, AssessmentEndSimpleMessage):
        return message_data.content, message_data.valid_responses

    if isinstance(message_data, AssessmentEndScoreBasedMessage):
        content_map = {
            "high": message_data.high_score_content,
            "medium": message_data.medium_score_content,
            "low": message_data.low_score_content,
            "skipped-many": message_data.skipped_many_content,
        }
        # Get the specific content block for the score
        score_content = content_map.get(
            score_category,
            AssessmentEndContentItem.model_validate(
                {"content": "", "valid_responses": []}
            ),
        )
        return score_content.content, score_content.valid_responses

    # This should ideally not be reached if type checking is correct
    raise TypeError("Unsupported message data type")


def determine_task(
    flow_id: str,
    previous_message_nr: int,
    score_category: str,
    processed_user_response: str,
) -> str:
    """
    Refactored Logic: Encapsulates the business logic for determining
    the background task.
    """
    if flow_id == "dma-pre-assessment":
        if (
            previous_message_nr == 1
            and score_category == "skipped-many"
            and processed_user_response == "Yes"
        ):
            return "REMIND_ME_LATER"
        if previous_message_nr == 2:
            return "STORE_FEEDBACK"

    if flow_id in ["behaviour-pre-assessment", "knowledge-pre-assessment"]:
        if previous_message_nr == 1 and processed_user_response == "Remind me tomorrow":
            return "REMIND_ME_LATER"

    if flow_id == "attitude-pre-assessment" and previous_message_nr == 1:
        if score_category == "skipped-many" and processed_user_response == "Yes":
            return "REMIND_ME_LATER"
        elif score_category != "skipped-many":
            return "STORE_FEEDBACK"

    return ""


def matches_assessment_question_length(
    n_questions: int, assessment_type: AssessmentType
):
    """
    Checks if the length of a list of AssessmentHistory objects matches the length of questions from the corresponding assessment.
    """
    if assessment_type.value == kab_b_post_flow_id:
        assessment_type = assessment_map_to_their_pre[kab_b_post_flow_id]
    print(len(assessment_flow_map[assessment_type.value]))
    print(n_questions)
    if len(assessment_flow_map[assessment_type.value]) == n_questions:
        return True
    return False
