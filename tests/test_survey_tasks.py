import pytest
from unittest import mock

from haystack.dataclasses import ChatMessage

from ai4gd_momconnect_haystack.tasks import (
    process_survey_turn,
    # We will also need to import our new helper functions and the context model
    # _initialize_turn_context,
    # _handle_turn_entry_points,
    # ... etc
)
from ai4gd_momconnect_haystack.pydantic_models import SurveyTurnContext, SurveyResponse

# --- Test Fixtures ---


@pytest.fixture
def mock_turn_context():
    """Provides a basic, reusable SurveyTurnContext object for tests."""
    return SurveyTurnContext(
        user_id="test-user",
        survey_id="anc",
        user_input="A regular answer",
        is_re_engagement_ping=False,
        failure_count=0,
        history=[ChatMessage.from_assistant("Last question?")],
        journey_state=None,
        last_assistant_message=ChatMessage.from_assistant("Last question?"),
        previous_context={},
        current_context={},
    )


# --- Tests for the Orchestrator and Helper Functions ---


@pytest.mark.asyncio
@mock.patch("ai4gd_momconnect_haystack.tasks._initialize_turn_context")
@mock.patch(
    "ai4gd_momconnect_haystack.tasks._handle_turn_entry_points", return_value=None
)
@mock.patch(
    "ai4gd_momconnect_haystack.tasks._process_active_response", return_value=None
)
@mock.patch("ai4gd_momconnect_haystack.tasks._conclude_turn")
async def test_process_survey_turn_follows_correct_pipeline(
    mock_conclude, mock_process, mock_entry_points, mock_initialize, mock_turn_context
):
    """
    Tests that the main orchestrator calls each step of the pipeline in the correct order for a standard turn.
    """
    # Arrange
    mock_initialize.return_value = mock_turn_context
    mock_conclude.return_value = SurveyResponse(
        question="Final question", user_context={}
    )

    # Act
    await process_survey_turn(
        user_id="test-user",
        survey_id="anc",
        user_input="A regular answer",
        user_context={},
        failure_count=0,
    )

    # Assert
    mock_initialize.assert_awaited_once()
    mock_entry_points.assert_awaited_once_with(mock_turn_context)
    mock_process.assert_awaited_once_with(mock_turn_context)
    mock_conclude.assert_awaited_once_with(mock_turn_context)


@pytest.mark.asyncio
@mock.patch("ai4gd_momconnect_haystack.tasks._resume_survey_from_re_engagement")
async def test_process_survey_turn_handles_re_engagement_ping(mock_resume):
    """
    Tests that a re-engagement ping is correctly routed and immediately handled.
    """
    # Arrange
    mock_resume.return_value = SurveyResponse(question="Welcome back!", user_context={})

    # Act
    response = await process_survey_turn(
        user_id="test-user",
        survey_id="anc",
        user_input=None,
        user_context={},
        failure_count=0,
        is_re_engagement_ping=True,
    )

    # Assert
    mock_resume.assert_awaited_once_with("test-user", "anc")
    assert response.question == "Welcome back!"


@pytest.mark.asyncio
@mock.patch("ai4gd_momconnect_haystack.tasks._handle_new_survey_turn")
async def test_process_survey_turn_handles_new_survey(mock_new_survey):
    """
    Tests that a turn with no user_input is correctly routed to the new survey handler.
    """
    # Arrange
    mock_new_survey.return_value = SurveyResponse(
        question="First question!", user_context={}
    )

    # Act
    response = await process_survey_turn(
        user_id="test-user",
        survey_id="anc",
        user_input="",  # Empty input signifies a new survey
        user_context={},
        failure_count=0,
    )

    # Assert
    mock_new_survey.assert_awaited_once()
    assert response.question == "First question!"


# We would continue to write focused unit tests for each helper function:
# - test__initialize_turn_context_loads_data_correctly
# - test__handle_turn_entry_points_routes_to_intro_handler
# - test__process_active_response_succeeds_on_extraction
# - test__process_active_response_falls_back_to_intent
# - test__process_active_response_triggers_repair
# - test__conclude_turn_calls_retriever_and_persists
# - test__persist_survey_turn_state_saves_correct_data
