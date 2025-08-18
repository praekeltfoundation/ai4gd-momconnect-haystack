from haystack.dataclasses import ChatMessage
from sqlalchemy.future import select

from ai4gd_momconnect_haystack.crud import (
    delete_assessment_history_for_user,
    delete_chat_history_for_user,
    get_assessment_history,
    get_or_create_chat_history,
    save_assessment_question,
    save_chat_history,
)
from ai4gd_momconnect_haystack.database import SessionLocal
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    AssessmentHistory,
    AssessmentResultHistory,
    ChatHistory,
)


def test_delete_chat_history_for_user_onboarding():
    """
    Test that delete_chat_history_for_user correctly deletes onboarding chat history
    for a specific user while leaving other history types intact.
    """
    user_id = "test_user_123"

    # Create test chat messages
    onboarding_messages = [
        ChatMessage.from_system("Welcome to onboarding"),
        ChatMessage.from_assistant("How are you today?"),
        ChatMessage.from_user("I'm doing well"),
    ]

    anc_messages = [
        ChatMessage.from_system("Welcome to ANC survey"),
        ChatMessage.from_assistant("When was your last visit?"),
        ChatMessage.from_user("Last week"),
    ]

    # Save both types of chat history
    save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)
    save_chat_history(user_id, anc_messages, HistoryType.anc)

    # Verify both histories exist before deletion
    onboarding_history_before = get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_before = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_before) == 3
    assert len(anc_history_before) == 3

    # Delete onboarding history
    delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify onboarding history is deleted but ANC history remains
    onboarding_history_after = get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 0
    assert len(anc_history_after) == 3
    assert anc_history_after[0].text == "Welcome to ANC survey"


def test_delete_chat_history_for_user_anc():
    """
    Test that delete_chat_history_for_user correctly deletes ANC chat history
    for a specific user while leaving onboarding history intact.
    """
    user_id = "test_user_456"

    # Create test chat messages
    onboarding_messages = [
        ChatMessage.from_system("Welcome to onboarding"),
        ChatMessage.from_assistant("What's your name?"),
        ChatMessage.from_user("My name is Alice"),
    ]

    anc_messages = [
        ChatMessage.from_system("ANC survey start"),
        ChatMessage.from_assistant("How many weeks pregnant are you?"),
        ChatMessage.from_user("20 weeks"),
    ]

    # Save both types of chat history
    save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)
    save_chat_history(user_id, anc_messages, HistoryType.anc)

    # Verify both histories exist before deletion
    onboarding_history_before = get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_before = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_before) == 3
    assert len(anc_history_before) == 3

    # Delete ANC history
    delete_chat_history_for_user(user_id, HistoryType.anc)

    # Verify ANC history is deleted but onboarding history remains
    onboarding_history_after = get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 3
    assert len(anc_history_after) == 0
    assert onboarding_history_after[0].text == "Welcome to onboarding"


def test_delete_chat_history_for_user_nonexistent_user():
    """
    Test that delete_chat_history_for_user handles gracefully when
    trying to delete history for a user that doesn't exist.
    """
    user_id = "nonexistent_user"

    # Attempt to delete history for a user that doesn't exist
    # This should not raise an exception
    delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify no history exists for this user
    history = get_or_create_chat_history(user_id, HistoryType.onboarding)
    assert len(history) == 0


def test_delete_chat_history_for_user_empty_history():
    """
    Test that delete_chat_history_for_user works correctly when
    the user has an empty history for the specified type.
    """
    user_id = "test_user_empty"

    # Create a user with only onboarding history, leaving ANC empty
    onboarding_messages = [
        ChatMessage.from_assistant("Hello"),
        ChatMessage.from_user("Hi there"),
    ]

    save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)

    # Verify initial state
    onboarding_history = get_or_create_chat_history(user_id, HistoryType.onboarding)
    anc_history = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history) == 2
    assert len(anc_history) == 0

    # Delete the empty ANC history (should work without error)
    delete_chat_history_for_user(user_id, HistoryType.anc)

    # Verify state is unchanged
    onboarding_history_after = get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 2
    assert len(anc_history_after) == 0


def test_delete_chat_history_database_state():
    """
    Test that delete_chat_history_for_user correctly updates the database
    by setting the appropriate history field to an empty list.
    """
    user_id = "test_user_db_state"

    # Create and save chat history
    messages = [
        ChatMessage.from_assistant("Test message"),
        ChatMessage.from_user("User response"),
    ]

    save_chat_history(user_id, messages, HistoryType.onboarding)

    # Verify the database record exists and has content
    with SessionLocal() as session:
        result = session.execute(
            select(ChatHistory).filter(ChatHistory.user_id == user_id)
        )
        db_history = result.scalar_one_or_none()
        assert db_history is not None
        assert len(db_history.onboarding_history) == 2
        assert len(db_history.anc_survey_history) == 0

    # Delete the onboarding history
    delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify the database record still exists but onboarding_history is empty
    with SessionLocal() as session:
        result = session.execute(
            select(ChatHistory).filter(ChatHistory.user_id == user_id)
        )
        db_history = result.scalar_one_or_none()
        assert db_history is not None
        assert len(db_history.onboarding_history) == 0
        assert len(db_history.anc_survey_history) == 0


def test_delete_assessment_history_for_user_basic():
    """
    Test that delete_assessment_history_for_user correctly deletes all assessment-related
    history for a specific user and assessment type.
    """
    user_id = "test_assessment_user_1"
    assessment_type = AssessmentType.dma_pre_assessment

    # Save some assessment questions
    save_assessment_question(user_id, assessment_type, 1, "Question 1", "Answer 1", 5)
    save_assessment_question(user_id, assessment_type, 2, "Question 2", "Answer 2", 3)
    save_assessment_question(user_id, assessment_type, 3, "Question 3", "Answer 3", 4)

    # Verify assessment history exists before deletion
    history_before = get_assessment_history(user_id, assessment_type)
    assert len(history_before) == 3

    # Verify records exist in the database
    with SessionLocal() as session:
        # Check AssessmentHistory
        result = session.execute(
            select(AssessmentHistory).filter(
                AssessmentHistory.user_id == user_id,
                AssessmentHistory.assessment_id == assessment_type.value,
            )
        )
        assessment_records = result.scalars().all()
        assert len(assessment_records) == 3

    # Delete assessment history
    delete_assessment_history_for_user(user_id, assessment_type)

    # Verify assessment history is deleted
    history_after = get_assessment_history(user_id, assessment_type)
    assert len(history_after) == 0

    # Verify database records are deleted
    with SessionLocal() as session:
        # Check AssessmentHistory
        result = session.execute(
            select(AssessmentHistory).filter(
                AssessmentHistory.user_id == user_id,
                AssessmentHistory.assessment_id == assessment_type.value,
            )
        )
        assessment_records = result.scalars().all()
        assert len(assessment_records) == 0

        # Check AssessmentResultHistory
        result = session.execute(
            select(AssessmentResultHistory).filter(
                AssessmentResultHistory.user_id == user_id,
                AssessmentResultHistory.assessment_id == assessment_type.value,
            )
        )
        result_records = result.scalars().all()
        assert len(result_records) == 0

        # Check AssessmentEndMessagingHistory
        result = session.execute(
            select(AssessmentEndMessagingHistory).filter(
                AssessmentEndMessagingHistory.user_id == user_id,
                AssessmentEndMessagingHistory.assessment_id == assessment_type.value,
            )
        )
        messaging_records = result.scalars().all()
        assert len(messaging_records) == 0


def test_delete_assessment_history_for_user_selective_deletion():
    """
    Test that delete_assessment_history_for_user only deletes history for the specified
    assessment type, leaving other assessment types intact.
    """
    user_id = "test_assessment_user_2"
    assessment_type_1 = AssessmentType.dma_pre_assessment
    assessment_type_2 = AssessmentType.knowledge_pre_assessment

    # Save assessment questions for both types
    save_assessment_question(
        user_id, assessment_type_1, 1, "DMA Question 1", "DMA Answer 1", 5
    )
    save_assessment_question(
        user_id, assessment_type_2, 1, "Knowledge Question 1", "Knowledge Answer 1", 3
    )

    # Verify both assessment histories exist
    history_1_before = get_assessment_history(user_id, assessment_type_1)
    history_2_before = get_assessment_history(user_id, assessment_type_2)
    assert len(history_1_before) == 1
    assert len(history_2_before) == 1

    # Delete only the first assessment type
    delete_assessment_history_for_user(user_id, assessment_type_1)

    # Verify only the first assessment type is deleted
    history_1_after = get_assessment_history(user_id, assessment_type_1)
    history_2_after = get_assessment_history(user_id, assessment_type_2)
    assert len(history_1_after) == 0
    assert len(history_2_after) == 1
    assert history_2_after[0].question == "Knowledge Question 1"


def test_delete_assessment_history_for_user_nonexistent():
    """
    Test that delete_assessment_history_for_user handles gracefully when
    trying to delete history for a user/assessment that doesn't exist.
    """
    user_id = "nonexistent_assessment_user"
    assessment_type = AssessmentType.attitude_pre_assessment

    # Attempt to delete history for a user/assessment that doesn't exist
    # This should not raise an exception
    delete_assessment_history_for_user(user_id, assessment_type)

    # Verify no history exists for this user/assessment
    history = get_assessment_history(user_id, assessment_type)
    assert len(history) == 0


def test_delete_assessment_history_for_user_user_isolation():
    """
    Test that delete_assessment_history_for_user only deletes history for the specified
    user, leaving other users' data intact.
    """
    user_id_1 = "test_assessment_user_3"
    user_id_2 = "test_assessment_user_4"
    assessment_type = AssessmentType.behaviour_pre_assessment

    # Save assessment questions for both users
    save_assessment_question(
        user_id_1, assessment_type, 1, "Question 1", "User 1 Answer", 5
    )
    save_assessment_question(
        user_id_2, assessment_type, 1, "Question 1", "User 2 Answer", 3
    )

    # Verify both users have assessment history
    history_1_before = get_assessment_history(user_id_1, assessment_type)
    history_2_before = get_assessment_history(user_id_2, assessment_type)
    assert len(history_1_before) == 1
    assert len(history_2_before) == 1

    # Delete only user 1's assessment history
    delete_assessment_history_for_user(user_id_1, assessment_type)

    # Verify only user 1's history is deleted
    history_1_after = get_assessment_history(user_id_1, assessment_type)
    history_2_after = get_assessment_history(user_id_2, assessment_type)
    assert len(history_1_after) == 0
    assert len(history_2_after) == 1
    assert history_2_after[0].user_response == "User 2 Answer"


def test_delete_assessment_history_database_cleanup():
    """
    Test that delete_assessment_history_for_user correctly removes all related
    records from AssessmentHistory, AssessmentResultHistory, and AssessmentEndMessagingHistory.
    """
    user_id = "test_assessment_cleanup_user"
    assessment_type = AssessmentType.dma_post_assessment

    # Create assessment question
    save_assessment_question(
        user_id, assessment_type, 1, "Test Question", "Test Answer", 4
    )

    # Manually create related records in the database
    with SessionLocal() as session:
        with session.begin():
            # Create AssessmentResultHistory record
            result_record = AssessmentResultHistory(
                user_id=user_id,
                assessment_id=assessment_type.value,
                category="test_category",
                score=75,
                crossed_skip_threshold=False,
            )
            session.add(result_record)

            # Create AssessmentEndMessagingHistory record
            messaging_record = AssessmentEndMessagingHistory(
                user_id=user_id,
                assessment_id=assessment_type.value,
                message_number=1,
                user_response="Test message response",
            )
            session.add(messaging_record)
            session.commit()

    # Verify all records exist before deletion
    with SessionLocal() as session:
        # Check AssessmentHistory
        result = session.execute(
            select(AssessmentHistory).filter(
                AssessmentHistory.user_id == user_id,
                AssessmentHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 1

        # Check AssessmentResultHistory
        result = session.execute(
            select(AssessmentResultHistory).filter(
                AssessmentResultHistory.user_id == user_id,
                AssessmentResultHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 1

        # Check AssessmentEndMessagingHistory
        result = session.execute(
            select(AssessmentEndMessagingHistory).filter(
                AssessmentEndMessagingHistory.user_id == user_id,
                AssessmentEndMessagingHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 1

    # Delete assessment history
    delete_assessment_history_for_user(user_id, assessment_type)

    # Verify all related records are deleted
    with SessionLocal() as session:
        # Check AssessmentHistory
        result = session.execute(
            select(AssessmentHistory).filter(
                AssessmentHistory.user_id == user_id,
                AssessmentHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 0

        # Check AssessmentResultHistory
        result = session.execute(
            select(AssessmentResultHistory).filter(
                AssessmentResultHistory.user_id == user_id,
                AssessmentResultHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 0

        # Check AssessmentEndMessagingHistory
        result = session.execute(
            select(AssessmentEndMessagingHistory).filter(
                AssessmentEndMessagingHistory.user_id == user_id,
                AssessmentEndMessagingHistory.assessment_id == assessment_type.value,
            )
        )
        assert len(result.scalars().all()) == 0
