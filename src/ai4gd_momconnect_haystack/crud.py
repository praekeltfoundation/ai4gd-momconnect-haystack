import logging
from typing import Any

from haystack.dataclasses import ChatMessage
from sqlalchemy import delete, update
from sqlalchemy.future import select

from ai4gd_momconnect_haystack.assessment_logic import (
    matches_assessment_question_length,
    score_assessment,
)
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentResult,
    AssessmentRun,
    Turn,
)
from ai4gd_momconnect_haystack.utilities import chat_messages_to_json

from .database import SessionLocal
from .sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    AssessmentHistory,
    AssessmentResultHistory,
    ChatHistory,
    UserJourneyState,
)

logger = logging.getLogger(__name__)


def get_or_create_chat_history(
    user_id: str, history_type: HistoryType
) -> list[ChatMessage]:
    """
    Retrieves an Onboarding chat history for a given user_id.
    If it doesn't exist, an empty history is returned.
    """
    with SessionLocal() as session:
        result = session.execute(
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
                    chat_history.append(
                        ChatMessage.from_user(text=cm["text"], meta=cm.get("meta", {}))
                    )
                elif cm["role"] == "assistant":
                    chat_history.append(
                        ChatMessage.from_assistant(
                            text=cm["text"], meta=cm.get("meta", {})
                        )
                    )
                else:
                    chat_history.append(
                        ChatMessage.from_system(
                            text=cm["text"], meta=cm.get("meta", {})
                        )
                    )
        return chat_history


def save_chat_history(
    user_id: str, messages: list[ChatMessage], history_type: HistoryType
) -> None:
    """Saves or updates the chat history for a given user_id."""
    with SessionLocal() as session:
        with session.begin():
            result = session.execute(
                select(ChatHistory).filter(ChatHistory.user_id == user_id)
            )
            db_history = result.scalar_one_or_none()

            history_json = chat_messages_to_json(messages)

            if db_history:
                if history_type.value == "onboarding":
                    db_history.onboarding_history = history_json
                elif history_type.value == "anc-survey":
                    db_history.anc_survey_history = history_json
                else:
                    logger.error(f"Unknown chat history type to update: {history_type}")
                    return
            else:
                if history_type.value == "onboarding":
                    db_history = ChatHistory(
                        user_id=user_id,
                        onboarding_history=history_json,
                        anc_survey_history=[],
                    )
                elif history_type.value == "anc-survey":
                    db_history = ChatHistory(
                        user_id=user_id,
                        onboarding_history=[],
                        anc_survey_history=history_json,
                    )
                else:
                    logger.error(f"Unknown chat history type to create: {history_type}")
                    return
                session.add(db_history)

            session.commit()


def get_assessment_history(
    user_id: str, assessment_type: AssessmentType
) -> list[AssessmentHistory]:
    """
    Retrieves assessment question history for a given user and assessment type.
    If no history exists, an empty list is returned.
    """
    with SessionLocal() as session:
        result = session.execute(
            select(AssessmentHistory).filter(
                AssessmentHistory.user_id == user_id,
                AssessmentHistory.assessment_id == assessment_type.value,
            )
        )
        history = result.scalars().all()
        return list(history)


def save_assessment_question(
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
    with SessionLocal() as session:
        with session.begin():
            result = session.execute(
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


def calculate_and_store_assessment_result(
    user_id: str, assessment_type: AssessmentType
) -> None:
    """
    Checks if an assessment is complete and, if so, calculates and stores the final result.
    """
    # Check if all scores are available for the given assessment
    with SessionLocal() as session:
        with session.begin():
            result = session.execute(
                select(AssessmentHistory)
                .where(
                    AssessmentHistory.user_id == user_id,
                    AssessmentHistory.assessment_id == assessment_type.value,
                    AssessmentHistory.score.is_not(None),
                )
                .order_by(AssessmentHistory.question_number.asc())
            )
            historic_records = result.scalars().all()

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
                        "llm_extracted_user_response": rec.user_response,
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
                session.execute(result_history_stmt)
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


def get_assessment_result(
    user_id: str, assessment_type: AssessmentType
) -> AssessmentResult | None:
    print(
        f"Trying to get assessment results for {user_id} for assessment {assessment_type.value}"
    )
    with SessionLocal() as session:
        with session.begin():
            result = session.execute(
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


def get_assessment_end_messaging_history(
    user_id: str, assessment_type: AssessmentType
) -> list[AssessmentEndMessagingHistory]:
    """
    Retrieves assessment end messaging history for a given user and assessment type.
    If no history exists, an empty list is returned.
    """
    with SessionLocal() as session:
        result = session.execute(
            select(AssessmentEndMessagingHistory)
            .filter(
                AssessmentEndMessagingHistory.user_id == user_id,
                AssessmentEndMessagingHistory.assessment_id == assessment_type.value,
            )
            .order_by(AssessmentEndMessagingHistory.message_number.asc())
        )
        history = result.scalars().all()
        return list(history)


def save_assessment_end_message(
    user_id: str,
    assessment_type: AssessmentType,
    message_number: int,
    user_response: str,
) -> None:
    """
    Saves the end-of-assessment messaging history for a given user, assessment type and question number.
    This will overwrite any existing history for the same user, assessment type and question number.
    """
    with SessionLocal() as session:
        with session.begin():
            # If a user_response is present, the question should already exist in the database and we can just update it.
            if user_response:
                session.execute(
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
                session.execute(
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
            session.commit()


def delete_chat_history_for_user(user_id: str, history_type: HistoryType):
    with SessionLocal() as session:
        with session.begin():
            result = session.execute(
                select(ChatHistory).filter(ChatHistory.user_id == user_id)
            )
            db_history = result.scalar_one_or_none()

            if db_history:
                if history_type == HistoryType.onboarding:
                    db_history.onboarding_history = []
                elif history_type == HistoryType.anc:
                    db_history.anc_survey_history = []
                else:
                    logger.error(f"Unknown chat history type to delete: {history_type}")
                    return


def delete_assessment_history_for_user(user_id: str, assessment_type: AssessmentType):
    with SessionLocal() as session:
        with session.begin():
            session.execute(
                delete(AssessmentHistory).where(
                    AssessmentHistory.user_id == user_id,
                    AssessmentHistory.assessment_id == assessment_type.value,
                )
            )
            session.execute(
                delete(AssessmentResultHistory).where(
                    AssessmentResultHistory.user_id == user_id,
                    AssessmentResultHistory.assessment_id == assessment_type.value,
                )
            )
            session.execute(
                delete(AssessmentEndMessagingHistory).where(
                    AssessmentEndMessagingHistory.user_id == user_id,
                    AssessmentEndMessagingHistory.assessment_id
                    == assessment_type.value,
                )
            )


def truncate_chat_history():
    with SessionLocal() as session:
        with session.begin():
            session.execute(delete(ChatHistory))


def truncate_assessment_history():
    with SessionLocal() as session:
        with session.begin():
            session.execute(delete(AssessmentHistory))


def truncate_assessment_result_history():
    with SessionLocal() as session:
        with session.begin():
            session.execute(delete(AssessmentResultHistory))


def truncate_assessment_end_messaging_history():
    with SessionLocal() as session:
        with session.begin():
            session.execute(delete(AssessmentEndMessagingHistory))


def save_user_journey_state(
    user_id: str,
    flow_id: str,
    step_identifier: str,
    last_question: str,
    user_context: dict[str, Any],
    reminder_type: int | None = None,
    next_question_number: int | None = None,
    expected_step_id: str | None = None,
):
    """
    Creates or updates the user's last known state in the database.
    This performs an 'upsert' operation.
    """
    if not last_question:
        logger.warning(
            f"Attempted to save journey state for user {user_id} with an empty last_question. Skipping."
        )
        return

    with SessionLocal() as session:
        with session.begin():
            # Try to get the existing state
            result = session.execute(
                select(UserJourneyState).where(UserJourneyState.user_id == user_id)
            )
            existing_state = result.scalar_one_or_none()

            if existing_state:
                # Update existing record
                existing_state.current_flow_id = flow_id
                existing_state.current_step_identifier = str(step_identifier)
                existing_state.last_question_sent = last_question
                existing_state.user_context = user_context
                existing_state.reminder_type = reminder_type
                if next_question_number:
                    existing_state.next_question_number = next_question_number
                if expected_step_id:
                    existing_state.expected_step_id = expected_step_id
                logger.info(f"Updated journey state for user {user_id}.")
            else:
                # Create new record
                new_state = UserJourneyState(
                    user_id=user_id,
                    current_flow_id=flow_id,
                    current_step_identifier=str(step_identifier),
                    last_question_sent=last_question,
                    user_context=user_context,
                    reminder_type=reminder_type,
                )
                if next_question_number:
                    new_state.next_question_number = next_question_number
                if expected_step_id:
                    new_state.expected_step_id = expected_step_id
                session.add(new_state)
                logger.info(f"Created new journey state for user {user_id}.")
            session.commit()


def get_user_journey_state(user_id: str) -> UserJourneyState | None:
    """Retrieves the last known journey state for a given user."""
    with SessionLocal() as session:
        result = session.execute(
            select(UserJourneyState).where(UserJourneyState.user_id == user_id)
        )
        return result.scalar_one_or_none()


def delete_user_journey_state(user_id: str) -> None:
    """Deletes the user's journey state from the database."""
    with SessionLocal() as session:
        with session.begin():
            session.execute(
                delete(UserJourneyState).where(UserJourneyState.user_id == user_id)
            )
            session.commit()
