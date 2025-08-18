# tests/test_survey_tasks.py

import pytest
from unittest import mock
from unittest.mock import AsyncMock

from haystack.dataclasses import ChatMessage

from ai4gd_momconnect_haystack.survey_orchestrator import process_survey_turn
from ai4gd_momconnect_haystack.pydantic_models import (
    OrchestratorSurveyRequest as SurveyRequest,
    OrchestratorUserJourneyState as UserJourneyState,
)
from ai4gd_momconnect_haystack.enums import Intent

# --- Test Fixtures for Reusable Test Data ---


@pytest.fixture
def mock_request() -> SurveyRequest:
    """Provides a default SurveyRequest object."""
    return SurveyRequest(
        user_id="test-user",
        survey_id="anc-survey",
        user_input="A regular answer",
        user_context={"first_survey": True},
        failure_count=0,
    )


@pytest.fixture
def mock_journey_state_intro() -> UserJourneyState:
    """Provides a UserJourneyState object for a user at the intro step."""
    return UserJourneyState(
        user_id="test-user",
        current_flow_id="anc-survey",
        current_step_identifier="intro",
        last_question_sent="Hi mom...?",
        user_context={"first_survey": True},
        version=1,
    )


@pytest.fixture
def mock_journey_state_active() -> UserJourneyState:
    """Provides a UserJourneyState for a user in the middle of the survey."""
    return UserJourneyState(
        user_id="test-user",
        current_flow_id="anc-survey",
        current_step_identifier="Q_seen",
        last_question_sent="Did a health worker see you?",
        user_context={"intro": "YES", "first_survey": True},
        version=2,
    )


# --- Tests for survey_orchestrator.py ---


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_new_survey_starts_correctly(
    mock_tasks: AsyncMock, mock_crud: AsyncMock, mock_request: SurveyRequest
):
    """
    Verifies that the first turn of a survey (no user_input) correctly
    fetches the intro message and sets the initial state.
    """
    # Arrange
    mock_request.user_input = ""
    mock_crud.get_user_journey_state.return_value = None
    mock_crud.get_or_create_chat_history.return_value = []

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question_identifier == "intro"
    assert not response.survey_complete
    mock_crud.save_user_journey_state.assert_awaited_once()
    saved_state_args = mock_crud.save_user_journey_state.call_args.kwargs
    assert saved_state_args["step_identifier"] == "intro"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_input, classified_as, expected_context_value, expected_next_step",
    [
        ("a", "YES", "YES", "Q_seen"),
        ("b", "NO", "NO", "Q_why_not_go"),
        ("c", "SOON", "SOON", "start_going_soon"),
    ],
    ids=["Yes response", "No response", "Soon response"],
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_intro_response_branching(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_intro: UserJourneyState,
    user_input,
    classified_as,
    expected_context_value,
    expected_next_step,
):
    """
    Verifies all three branches of the intro question.
    """
    # Arrange
    mock_request.user_input = user_input
    mock_crud.get_user_journey_state.return_value = mock_journey_state_intro
    mock_crud.get_or_create_chat_history.return_value = [
        ChatMessage.from_assistant("...")
    ]

    # Configure the synchronous classifier to be a standard mock
    mock_tasks.classify_anc_start_response = mock.Mock(return_value=classified_as)

    # The get_anc_survey_question function is async, so its mock setup is correct
    mock_tasks.get_anc_survey_question.return_value = {
        "question_identifier": expected_next_step,
        "contextualized_question": "Next question text",
    }

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question_identifier == expected_next_step
    mock_tasks.get_anc_survey_question.assert_awaited_once()
    call_context = mock_tasks.get_anc_survey_question.call_args.kwargs["user_context"]
    assert call_context.get("intro") == expected_context_value


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_active_turn_successful_extraction(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_active: UserJourneyState,
):
    """
    Tests the "happy path" for a standard turn where the user provides a valid
    response that is successfully extracted by the LLM.
    """
    # Arrange
    mock_request.user_input = "yes, I was"
    mock_crud.get_user_journey_state.return_value = mock_journey_state_active
    mock_crud.get_or_create_chat_history.return_value = [
        ChatMessage.from_assistant("...")
    ]

    updated_context = mock_journey_state_active.user_context.copy()
    updated_context["Q_seen"] = "YES"

    # Configure synchronous functions to be standard mocks
    mock_tasks.classify_yes_no_response = mock.Mock(return_value="AFFIRMATIVE")

    mock_tasks.get_anc_survey_question.return_value = {
        "question_identifier": "seen_yes",
        "contextualized_question": "Next question text",
    }

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question_identifier == "seen_yes"
    mock_tasks.get_anc_survey_question.assert_awaited_once()
    call_context = mock_tasks.get_anc_survey_question.call_args.kwargs["user_context"]
    assert call_context.get("Q_seen") == "YES"


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_active_turn_extraction_failure_triggers_repair(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_active: UserJourneyState,
):
    """
    Verifies that if data extraction fails, the system correctly falls back to repair.
    """
    # Arrange
    mock_request.user_input = "i dont know"
    mock_crud.get_user_journey_state.return_value = mock_journey_state_active
    mock_crud.get_or_create_chat_history.return_value = [
        ChatMessage.from_assistant("...")
    ]

    # Configure synchronous functions to be standard mocks
    mock_tasks.classify_yes_no_response = mock.Mock(return_value="AMBIGUOUS")
    mock_tasks.extract_anc_data_from_response = mock.Mock(
        return_value=(mock_journey_state_active.user_context, None)
    )
    mock_tasks.handle_user_message = mock.Mock(
        return_value=(Intent.CHITCHAT.value, None)
    )
    mock_tasks.handle_conversational_repair = mock.Mock(
        return_value="Sorry, please try again."
    )

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question == "Sorry, please try again."
    assert response.intent == "REPAIR"
    assert response.failure_count == 1


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_system_skip_after_max_failures(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_active: UserJourneyState,
):
    """
    Tests the n-strike policy: after reaching max failures, the question is force-skipped.
    """
    # Arrange
    mock_request.failure_count = 1
    mock_crud.get_user_journey_state.return_value = mock_journey_state_active
    mock_crud.get_or_create_chat_history.return_value = [
        ChatMessage.from_assistant("...")
    ]

    # Configure synchronous functions to be standard mocks
    mock_tasks.classify_yes_no_response = mock.Mock(return_value="AMBIGUOUS")
    mock_tasks.extract_anc_data_from_response = mock.Mock(
        return_value=(mock_journey_state_active.user_context, None)
    )
    mock_tasks.handle_user_message = mock.Mock(
        return_value=(Intent.CHITCHAT.value, None)
    )

    mock_tasks.get_anc_survey_question.return_value = {
        "question_identifier": "seen_yes",
        "contextualized_question": "Next question text",
    }

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question_identifier == "seen_yes"
    mock_tasks.get_anc_survey_question.assert_awaited_once()
    call_context = mock_tasks.get_anc_survey_question.call_args.kwargs["user_context"]
    assert call_context.get("Q_seen") == "Skipped - System"
    assert response.failure_count == 0


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_end_of_survey_is_handled(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_active: UserJourneyState,
):
    """
    Verifies that when the logic map returns None, the survey is marked as complete.
    """
    # Arrange
    mock_request.user_input = "Final answer"
    mock_journey_state_active.current_step_identifier = "end_if_feedback"
    mock_crud.get_user_journey_state.return_value = mock_journey_state_active
    mock_crud.get_or_create_chat_history.return_value = [
        ChatMessage.from_assistant("...")
    ]

    # ✅ FIX: Use side_effect to correctly mimic the real function's behavior.
    # This small function now modifies the user_context it receives, just like the real code.
    def mock_extraction(
        user_response, user_context, step_title, contextualized_question
    ):
        user_context["end_if_feedback"] = "some feedback"
        return user_context, None

    mock_tasks.extract_anc_data_from_response = mock.Mock(side_effect=mock_extraction)

    # This is the key mock that tells the orchestrator the survey is over after a valid turn
    mock_tasks.get_anc_survey_question.return_value = None

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.survey_complete is True
    assert response.question is not None
    assert "Thank you" in response.question


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_re_engagement_ping_resumes_correctly(
    mock_tasks: AsyncMock,
    mock_crud: AsyncMock,
    mock_request: SurveyRequest,
    mock_journey_state_active: UserJourneyState,
):
    """
    Verifies that a re-engagement ping correctly resumes the user
    at their last known question.
    """
    # Arrange
    mock_request.is_re_engagement_ping = True
    mock_request.user_input = None
    mock_crud.get_user_journey_state.return_value = mock_journey_state_active
    mock_crud.get_or_create_chat_history.return_value = []

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question is not None
    assert "Did a health worker see you?" in response.question
    assert response.question_identifier == "Q_seen"
    assert response.intent == Intent.JOURNEY_RESUMED.value
    mock_tasks.extract_anc_data_from_response.assert_not_called()
    mock_tasks.get_anc_survey_question.assert_not_called()


@pytest.mark.asyncio
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.crud", new_callable=AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.survey_orchestrator.tasks", new_callable=AsyncMock
)
async def test_unexpected_error_returns_safe_response(
    mock_tasks: AsyncMock, mock_crud: AsyncMock, mock_request: SurveyRequest
):
    """
    Verifies that if a dependency raises an unexpected exception, the
    orchestrator catches it and returns a generic, safe error message.
    """
    # Arrange
    mock_crud.get_user_journey_state.side_effect = Exception(
        "Database connection failed"
    )

    # Act
    response = await process_survey_turn(mock_request)

    # Assert
    assert response.question is not None
    assert "Sorry, we’ve run into a technical problem" in response.question
    assert response.intent == Intent.SYSTEM_ERROR.value
    assert response.survey_complete is True
