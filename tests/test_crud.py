import pytest
from haystack.dataclasses import ChatMessage
from sqlalchemy.future import select

from ai4gd_momconnect_haystack.crud import (
    delete_chat_history_for_user,
    get_or_create_chat_history,
    save_chat_history,
)
from ai4gd_momconnect_haystack.database import AsyncSessionLocal
from ai4gd_momconnect_haystack.enums import HistoryType
from ai4gd_momconnect_haystack.sqlalchemy_models import ChatHistory


@pytest.mark.asyncio
async def test_delete_chat_history_for_user_onboarding():
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
    await save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)
    await save_chat_history(user_id, anc_messages, HistoryType.anc)

    # Verify both histories exist before deletion
    onboarding_history_before = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_before = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_before) == 3
    assert len(anc_history_before) == 3

    # Delete onboarding history
    await delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify onboarding history is deleted but ANC history remains
    onboarding_history_after = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 0
    assert len(anc_history_after) == 3
    assert anc_history_after[0].text == "Welcome to ANC survey"


@pytest.mark.asyncio
async def test_delete_chat_history_for_user_anc():
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
    await save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)
    await save_chat_history(user_id, anc_messages, HistoryType.anc)

    # Verify both histories exist before deletion
    onboarding_history_before = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_before = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_before) == 3
    assert len(anc_history_before) == 3

    # Delete ANC history
    await delete_chat_history_for_user(user_id, HistoryType.anc)

    # Verify ANC history is deleted but onboarding history remains
    onboarding_history_after = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 3
    assert len(anc_history_after) == 0
    assert onboarding_history_after[0].text == "Welcome to onboarding"


@pytest.mark.asyncio
async def test_delete_chat_history_for_user_nonexistent_user():
    """
    Test that delete_chat_history_for_user handles gracefully when
    trying to delete history for a user that doesn't exist.
    """
    user_id = "nonexistent_user"

    # Attempt to delete history for a user that doesn't exist
    # This should not raise an exception
    await delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify no history exists for this user
    history = await get_or_create_chat_history(user_id, HistoryType.onboarding)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_delete_chat_history_for_user_empty_history():
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

    await save_chat_history(user_id, onboarding_messages, HistoryType.onboarding)

    # Verify initial state
    onboarding_history = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history) == 2
    assert len(anc_history) == 0

    # Delete the empty ANC history (should work without error)
    await delete_chat_history_for_user(user_id, HistoryType.anc)

    # Verify state is unchanged
    onboarding_history_after = await get_or_create_chat_history(
        user_id, HistoryType.onboarding
    )
    anc_history_after = await get_or_create_chat_history(user_id, HistoryType.anc)

    assert len(onboarding_history_after) == 2
    assert len(anc_history_after) == 0


@pytest.mark.asyncio
async def test_delete_chat_history_database_state():
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

    await save_chat_history(user_id, messages, HistoryType.onboarding)

    # Verify the database record exists and has content
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatHistory).filter(ChatHistory.user_id == user_id)
        )
        db_history = result.scalar_one_or_none()
        assert db_history is not None
        assert len(db_history.onboarding_history) == 2
        assert len(db_history.anc_survey_history) == 0

    # Delete the onboarding history
    await delete_chat_history_for_user(user_id, HistoryType.onboarding)

    # Verify the database record still exists but onboarding_history is empty
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatHistory).filter(ChatHistory.user_id == user_id)
        )
        db_history = result.scalar_one_or_none()
        assert db_history is not None
        assert len(db_history.onboarding_history) == 0
        assert len(db_history.anc_survey_history) == 0
