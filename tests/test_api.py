import os
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from haystack.dataclasses import ChatMessage
from sentry_sdk import get_client as get_sentry_client

from ai4gd_momconnect_haystack.api import app, setup_sentry, survey
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndScoreBasedMessage,
    AssessmentResult,
    SurveyResponse,
    ReengagementInfo,
    SurveyRequest,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    UserJourneyState,
)


from ai4gd_momconnect_haystack.database import AsyncSessionLocal

SERVICE_PERSONA_TEXT = "Test Persona"


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_onboarding_missing_auth():
    client = TestClient(app)
    response = client.post("/v1/onboarding")
    assert response.status_code == 422


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_onboarding_invalid_auth_value():
    client = TestClient(app)
    response = client.post("/v1/onboarding", headers={"Authorization": ""})
    assert response.status_code == 401


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_onboarding_invalid_auth_scheme():
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding", headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "endpoint", ["/v1/onboarding", "/v1/assessment", "/v1/survey", "/v1/resume"]
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_invalid_auth(endpoint, client: TestClient):
    response = client.post(endpoint, headers={"Authorization": "Bearer testtoken"})
    assert response.status_code == 401
    response = client.post(endpoint, headers={"Authorization": "Token invalid"})
    assert response.status_code == 401


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_onboarding_invalid_auth_token():
    client = TestClient(app)
    response = client.post("/v1/onboarding", headers={"Authorization": "Token invalid"})
    assert response.status_code == 401


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding_chitchat(
    mock_get_history, mock_handle_user_message, mock_repair
):
    """
    Tests that when a user provides a chitchat response during the onboarding flow,
    the system triggers the conversational repair mechanism instead of proceeding.
    """
    initial_history = [
        ChatMessage.from_system("..."),
        ChatMessage.from_assistant("Intro message..."),
        ChatMessage.from_user("Yes"),  # User has already consented
        ChatMessage.from_assistant("Original question?"),
    ]
    mock_get_history.return_value = initial_history
    mock_handle_user_message.return_value = ("CHITCHAT", "User is chitchatting")
    mock_repair.return_value = "Rephrased question."

    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "Hello!",
            "failure_count": 0,
        },
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Rephrased question."
    assert json_response["intent"] == "REPAIR"
    assert json_response["failure_count"] == 1
    mock_repair.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA_TEXT", "Test Persona")
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_chat_history_for_user",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding_first_question(
    get_or_create_chat_history,
    delete_chat_history_for_user,
    save_chat_history,
    get_next_onboarding_question,
):
    """
    For the first interaction (no user input), the API should return the
    introduction message and correctly initialize the chat history.
    """
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "TestUser", "user_context": {}, "user_input": ""},
    )

    # 1. Assert the API response is correct
    assert response.status_code == 200
    json_response = response.json()
    assert "Shall we begin?" in json_response["question"]
    assert json_response["intent"] == "SYSTEM_INTRO"

    # 2. Assert that history creation was attempted first
    get_or_create_chat_history.assert_awaited_once_with(
        user_id="TestUser", history_type=HistoryType.onboarding
    )

    # 3. Assert that the old history was then deleted
    delete_chat_history_for_user.assert_awaited_once_with(
        "TestUser", HistoryType.onboarding
    )

    # 4. Assert that the new history is saved correctly
    save_chat_history.assert_awaited_once_with(
        user_id="TestUser", messages=mock.ANY, history_type=HistoryType.onboarding
    )
    saved_messages = save_chat_history.call_args.kwargs["messages"]

    # 5. Assert the content of the saved history
    assert len(saved_messages) == 2
    assert saved_messages[0].is_from(role="system")
    assert saved_messages[0].text == "Test Persona"
    assert saved_messages[1].is_from(role="assistant")
    assert "Shall we begin?" in saved_messages[1].text

    # 6. Assert that the regular logic to get the next question was NOT called
    get_next_onboarding_question.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding(
    get_or_create_chat_history,
    save_chat_history,
    handle_user_message,
    mock_process_step,
):
    """
    A standard user response (after consent) should be processed,
    context updated, and the next question returned.
    """
    initial_history = [
        ChatMessage.from_system(SERVICE_PERSONA_TEXT),
        ChatMessage.from_assistant("Intro message..."),
        ChatMessage.from_user("Yes"),
        ChatMessage.from_assistant("Welcome!"),
    ]
    get_or_create_chat_history.return_value = initial_history
    mock_process_step.return_value = (
        {"area_type": "City"},
        {"contextualized_question": "Next Q"},
    )

    client = TestClient(app)
    client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "TestUser", "user_context": {}, "user_input": "city"},
    )
    saved_messages = save_chat_history.call_args.kwargs["messages"]
    assert len(saved_messages) == 6
    assert saved_messages[4].text == "city"
    mock_process_step.assert_called_once_with(
        user_input="city", current_context={}, current_question="Welcome!"
    )


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
async def test_assessment_chitchat(mock_handle_user_message, mock_repair):
    """
    Tests that a chitchat response during an assessment correctly triggers the
    conversational repair flow, re-phrasing the question rather than attempting
    to validate the answer.
    """
    mock_handle_user_message.return_value = ("CHITCHAT", "User is chitchatting")
    mock_repair.return_value = "Rephrased assessment question."

    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "Hello!",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "How are you feeling?",
            "failure_count": 0,
        },
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Rephrased assessment question."
    assert json_response["intent"] == "REPAIR"
    assert json_response["failure_count"] == 1
    mock_repair.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_assessment_history_for_user",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
async def test_assessment_initial_message(
    save_assessment_question,
    validate_assessment_answer,
    delete_assessment_history_for_user,
    get_assessment_question,
    handle_user_message,
):
    """
    On the first interaction with the user, we don't have a user input. This should be
    treated as a Journey Response and we should not check the intent or extract an answer.
    """
    get_assessment_question.return_value = {
        "contextualized_question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "current_question_number": 1,
    }

    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "next_question": 1,
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "processed_answer": None,
        "failure_count": 0,
        "reengagement_info": None,
    }

    handle_user_message.assert_not_called()
    validate_assessment_answer.assert_not_called()
    delete_assessment_history_for_user.assert_awaited_once_with(
        "TestUser", AssessmentType.dma_pre_assessment
    )
    get_assessment_question.assert_awaited_once()
    save_assessment_question.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.calculate_and_store_assessment_result",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch("ai4gd_momconnect_haystack.api.score_assessment_question")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
async def test_assessment_valid_journey_response(
    save_assessment_question,
    score_assessment_question,
    validate_assessment_answer,
    calculate_and_store_assessment_result,
    get_assessment_question,
    handle_user_message,
):
    """
    On a valid user journey response, the endpoint should validate the answer, score it,
    save it, calculate the aggregate result, and then fetch and return the
    next question, ensuring `failure_count` is correctly included in the final response.
    """
    # --- Mock Setup ---
    # 1. Intent detection identifies a standard journey response.
    handle_user_message.return_value = "JOURNEY_RESPONSE", ""

    # 2. Validation is successful, processes the answer, and points to question 2.
    validate_assessment_answer.return_value = {
        "processed_user_response": "very confident",
        "next_question_number": 2,
    }

    # 3. Scoring returns a score for the processed answer.
    score_assessment_question.return_value = 5

    # 4. The next question (question 2) is fetched.
    get_assessment_question.return_value = {
        "contextualized_question": "This is the second question.",
        "current_question_number": 2,
    }

    # --- API Call ---
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "I feel very confident",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "How confident are you?",
            "failure_count": 0,
        },
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "question": "This is the second question.",
        "next_question": 2,
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "processed_answer": "very confident",
        "failure_count": 0,
        "reengagement_info": None,
    }

    # Assert that the core logic functions were called with the correct arguments
    handle_user_message.assert_called_once_with(
        "How confident are you?", "I feel very confident"
    )
    validate_assessment_answer.assert_called_once_with(
        user_response="I feel very confident",
        question_number=1,
        current_flow_id="dma-pre-assessment",
    )
    score_assessment_question.assert_called_once_with(
        "very confident", 1, "dma-pre-assessment"
    )

    calculate_and_store_assessment_result.assert_awaited_once_with(
        "TestUser", "dma-pre-assessment"
    )

    get_assessment_question.assert_awaited_once_with(
        user_id="TestUser",
        flow_id="dma-pre-assessment",
        question_number=2,  # Importantly, it asks for the *next* question
        user_context={},
    )

    # Assert that the data was saved correctly in two separate calls
    assert save_assessment_question.await_count == 2

    # Check the call to save the user's answer and score
    call_to_save_answer = mock.call(
        user_id="TestUser",
        assessment_type="dma-pre-assessment",
        question_number=1,
        question="How confident are you?",
        user_response="very confident",
        score=5,
    )
    # Check the call to save the new question that was asked
    call_to_save_question = mock.call(
        user_id="TestUser",
        assessment_type="dma-pre-assessment",
        question_number=2,
        question="This is the second question.",
        user_response=None,
        score=None,
    )

    save_assessment_question.assert_has_awaits(
        [call_to_save_answer, call_to_save_question], any_order=False
    )


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch(
    "ai4gd_momconnect_haystack.api.assessment_end_flow_map",
    {
        "dma-pre-assessment": [
            AssessmentEndScoreBasedMessage.model_validate(
                {
                    "message_nr": 1,
                    "high-score-content": {"content": "High score content"},
                    "medium-score-content": {"content": "Medium score content"},
                    "low-score-content": {"content": "Low score content"},
                    "skipped-many-content": {"content": "Skipped many content"},
                }
            )
        ]
    },
)
async def test_assessment_end_initial_message(
    mock_get_content,
    mock_save_message,
    mock_get_history,
    mock_get_result,
):
    """
    Tests the initial call to the endpoint with no user_input.
    It should return the first message of the assessment-end flow.
    """
    # --- Mock Setup ---
    mock_get_result.return_value = AssessmentResult(
        score=100.0, category="high", crossed_skip_threshold=False
    )
    mock_get_history.return_value = []
    mock_get_content.return_value = ("High score content", None)

    # --- API Call ---
    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_input": "",
            "flow_id": "dma-pre-assessment",
        },
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "message": "High score content",
        "task": "",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "reengagement_info": None,
    }

    mock_get_result.assert_awaited_once()
    mock_get_history.assert_awaited_once()
    mock_save_message.assert_awaited_once_with("TestUser", "dma-pre-assessment", 1, "")


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_end_response")
@mock.patch("ai4gd_momconnect_haystack.api.determine_task")
@mock.patch("ai4gd_momconnect_haystack.api.assessment_end_flow_map")
@mock.patch("ai4gd_momconnect_haystack.api.response_is_required_for")
async def test_assessment_end_valid_response_to_required_question(
    mock_response_required,
    mock_flow_map,
    mock_determine_task,
    mock_validate_response,
    mock_handle_user_message,
    mock_get_content,
    mock_save_message,
    mock_get_history,
    mock_get_result,
):
    """
    Tests a valid user response to a message that requires validation.
    The system should process the answer, save it, and return the next message.
    """
    # --- Mock Setup ---
    mock_get_result.return_value = AssessmentResult(
        score=100.0, category="high", crossed_skip_threshold=False
    )
    mock_get_history.return_value = [
        AssessmentEndMessagingHistory(message_number=2, user_response="")
    ]
    mock_response_required.side_effect = lambda flow, nr: nr == 2

    # Corrected keys to use aliases from the Pydantic model
    flow_content = [
        AssessmentEndScoreBasedMessage.model_validate(
            {
                "message_nr": 2,
                "high-score-content": {
                    "content": "Would you like a summary? (Yes/No)",
                    "valid_responses": ["Yes", "No"],
                },
                "medium-score-content": {"content": "placeholder"},
                "low-score-content": {"content": "placeholder"},
                "skipped-many-content": {"content": "placeholder"},
            }
        ),
        AssessmentEndScoreBasedMessage.model_validate(
            {
                "message_nr": 3,
                "high-score-content": {"content": "Thank you for your feedback."},
                "medium-score-content": {"content": "placeholder"},
                "low-score-content": {"content": "placeholder"},
                "skipped-many-content": {"content": "placeholder"},
            }
        ),
    ]
    mock_flow_map.__getitem__.return_value = flow_content

    mock_get_content.side_effect = [
        ("Would you like a summary? (Yes/No)", ["Yes", "No"]),
        ("Thank you for your feedback.", None),
    ]
    mock_handle_user_message.return_value = ("JOURNEY_RESPONSE", "")
    mock_validate_response.return_value = {
        "processed_user_response": "Yes",
        "next_message_number": 3,
    }
    mock_determine_task.return_value = "SEND_SUMMARY"

    # --- API Call ---
    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_input": "Yes please",
            "flow_id": "dma-pre-assessment",
        },
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "message": "Thank you for your feedback.",
        "task": "SEND_SUMMARY",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "reengagement_info": None,
    }

    mock_validate_response.assert_called_once()
    mock_determine_task.assert_called_once()
    assert mock_save_message.await_count == 2
    call_to_save_answer = mock.call("TestUser", "dma-pre-assessment", 2, "Yes")
    call_to_save_question = mock.call("TestUser", "dma-pre-assessment", 3, "")
    mock_save_message.assert_has_awaits(
        [call_to_save_answer, call_to_save_question], any_order=True
    )


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_end_response")
@mock.patch("ai4gd_momconnect_haystack.api.assessment_end_flow_map")
@mock.patch("ai4gd_momconnect_haystack.api.response_is_required_for")
async def test_assessment_end_invalid_response(
    mock_response_required,
    mock_flow_map,
    mock_validate_response,
    mock_handle_user_message,
    mock_get_content,
    mock_save_message,
    mock_get_history,
    mock_get_result,
):
    """
    Tests an invalid user response to a required question.
    The system should repeat the question.
    """
    # --- Mock Setup ---
    mock_get_result.return_value = AssessmentResult(
        score=100.0, category="high", crossed_skip_threshold=False
    )
    mock_get_history.return_value = [
        AssessmentEndMessagingHistory(message_number=2, user_response="")
    ]
    mock_response_required.return_value = True

    # Corrected keys to use aliases from the Pydantic model
    flow_content = [
        AssessmentEndScoreBasedMessage.model_validate(
            {
                "message_nr": 2,
                "high-score-content": {
                    "content": "Would you like a summary? (Yes/No)",
                    "valid_responses": ["Yes", "No"],
                },
                "medium-score-content": {"content": "placeholder"},
                "low-score-content": {"content": "placeholder"},
                "skipped-many-content": {"content": "placeholder"},
            }
        )
    ]
    mock_flow_map.__getitem__.return_value = flow_content

    mock_get_content.side_effect = [
        ("Would you like a summary? (Yes/No)", ["Yes", "No"]),
        ("Would you like a summary? (Yes/No)", ["Yes", "No"]),
    ]
    mock_handle_user_message.return_value = ("JOURNEY_RESPONSE", "")
    mock_validate_response.return_value = {
        "processed_user_response": None,
        "next_message_number": 2,
    }

    # --- API Call ---
    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_input": "Maybe, I'm not sure",
            "flow_id": "dma-pre-assessment",
        },
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json()["message"] == "Would you like a summary? (Yes/No)"
    assert response.json()["task"] == ""

    mock_validate_response.assert_called_once()
    mock_save_message.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_anc_survey_question", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.extract_anc_data_from_response"
)  # This is the key mock to fix
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_anc_survey(
    mock_get_history, mock_save_history, mock_handle_msg, mock_extract, mock_get_q
):
    """
    Tests a standard, successful turn in the ANC survey after the intro is complete.
    """
    initial_history = [
        ChatMessage.from_system("..."),
        ChatMessage.from_assistant("Intro", meta={"step_title": "intro"}),
        ChatMessage.from_user("Yes"),
        ChatMessage.from_assistant("Q1", meta={"step_title": "start"}),
    ]
    mock_get_history.return_value = initial_history
    mock_handle_msg.return_value = ("JOURNEY_RESPONSE", None)

    mock_extract.return_value = ({"visit_status": "Yes, I went"}, None)

    mock_get_q.return_value = {
        "contextualized_question": "Q2",
        "question_identifier": "next_step",
        "is_final_step": False,
    }

    client = TestClient(app)
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "Yes I did",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    mock_save_history.assert_awaited_once()
    saved_messages = mock_save_history.call_args.kwargs["messages"]
    assert len(saved_messages) == 6
    assert saved_messages[4].text == "Yes I did"
    assert saved_messages[5].text == "Q2"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA_TEXT", "Test Persona")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_anc_survey_question",
    return_value={
        "contextualized_question": "Hi! Did you go for your clinic visit?",
        "is_final_step": False,
        "question_identifier": "start",
    },
)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_intro_response",
    return_value={
        "action": "PROCEED",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    },
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_anc_survey_first_question(
    get_or_create_chat_history,
    save_chat_history,
    mock_handle_intro,
    mock_handle_user_message,
    mock_get_question,
):
    """
    Tests that after a user provides consent, they receive the first proper
    ANC survey question.
    """
    history_after_intro_was_sent = [
        ChatMessage.from_system(text="Test Persona"),
        ChatMessage.from_assistant(
            text="The intro message.", meta={"step_title": "intro"}
        ),
    ]
    get_or_create_chat_history.return_value = history_after_intro_was_sent

    client = TestClient(app)
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "Yes",  # User gives consent
        },
    )

    assert response.status_code == 200
    assert response.json()["question"] == "Hi! Did you go for your clinic visit?"
    get_or_create_chat_history.assert_awaited_once_with(
        user_id="TestUser", history_type=HistoryType.anc
    )
    mock_handle_intro.assert_called_once_with(user_input="Yes", flow_id="anc-survey")
    mock_get_question.assert_awaited_once()
    mock_handle_user_message.assert_not_called()
    save_chat_history.assert_awaited_once()
    saved_messages = save_chat_history.call_args.kwargs["messages"]
    assert len(saved_messages) == 4
    assert saved_messages[2].text == "Yes"
    assert saved_messages[3].text == "Hi! Did you go for your clinic visit?"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch.dict(
    "ai4gd_momconnect_haystack.api.ANC_SURVEY_MAP",
    {"clinic_visit_prompt": mock.Mock(valid_responses=["Yes", "No"])},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_conversational_repair",
    return_value="Rephrased question.",
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_anc_survey_question",
    return_value={
        "contextualized_question": "Hi! Did you go for your clinic visit?",
        "is_final_step": False,
        "question_identifier": "clinic_visit_prompt",
    },
)
@mock.patch("ai4gd_momconnect_haystack.api.extract_anc_data_from_response")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("CHITCHAT", "User is chitchatting"),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history", new_callable=mock.AsyncMock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_anc_survey_chitchat(
    mock_get_history,
    mock_save_history,
    mock_handle_message,
    mock_extract_data,
    mock_get_question,
    mock_repair,
):
    """
    If the user sends chitchat during a survey, the API should trigger a
    conversational repair.
    """
    initial_history = [
        ChatMessage.from_assistant(
            "Hi! Did you go for your clinic visit?",
            meta={"step_title": "clinic_visit_prompt"},
        )
    ]
    mock_get_history.return_value = initial_history

    client = TestClient(app)
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "Hi!",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Rephrased question."
    assert json_response["intent"] == "REPAIR"
    mock_handle_message.assert_called_once_with(
        "Hi! Did you go for your clinic visit?", "Hi!", ["Yes", "No"]
    )


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_survey_invalid_survey_id():
    """
    If the user sends an unrecognised ID, then we should return an error message
    """
    client = TestClient(app)
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "survey_id": "invalid",
            "user_context": {},
            "user_input": "Hi",
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "ctx": {"expected": "'anc' or 'onboarding'"},
                "input": "invalid",
                "loc": ["body", "survey_id"],
                "msg": "Input should be 'anc' or 'onboarding'",
                "type": "enum",
            }
        ]
    }


@mock.patch.dict(
    os.environ,
    {"SENTRY_DSN": "https://testdsn@testdsn.example.org/12345"},
    clear=True,
)
def test_sentry_setup():
    """
    Sentry is setup with the correct DSN found in the env var
    """
    setup_sentry()
    assert get_sentry_client().is_active()
    assert get_sentry_client().dsn == "https://testdsn@testdsn.example.org/12345"


def test_prometheus_metrics():
    """
    Prometheus metrics are exposed on the /metrics endpoint
    """
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_info" in response.text


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_assessment_history_for_user",
    new_callable=mock.AsyncMock,
)
def test_assessment_initial_message_with_intro(mock_delete_history):
    """
    Tests that a flow configured to have an intro (e.g., behaviour)
    receives the intro message on the first call.
    """
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "",
            "flow_id": "behaviour-pre-assessment",
            "question_number": 1,
            "previous_question": "",
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert "Shall we begin?" in json_response["question"]
    assert json_response["next_question"] == 0
    assert json_response["intent"] == "SYSTEM_INTRO"
    mock_delete_history.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_assessment_history_for_user",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
async def test_assessment_initial_message_skips_intro(
    mock_get_q, mock_save_q, mock_delete_history
):
    """
    Tests that an assessment flow NOT configured to have an intro message skips the
    consent step and proceeds directly to fetching and returning the first question.
    """
    mock_get_q.return_value = {"contextualized_question": "Question 1"}
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "",
            "failure_count": 0,
        },
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Question 1"
    assert json_response["next_question"] == 1
    assert json_response["intent"] == "JOURNEY_RESPONSE"
    mock_delete_history.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.handle_intro_response")
async def test_assessment_consent_proceeds(
    mock_handle_intro, mock_handle_user_message, mock_get_q, mock_save_q
):
    """
    Tests that after an intro message is sent, a positive consent from the user ('Yes')
    correctly advances the flow to the first actual question of the assessment.
    """
    mock_handle_intro.return_value = {
        "action": "PROCEED",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    }
    mock_handle_user_message.return_value = ("JOURNEY_RESPONSE", "")
    mock_get_q.return_value = {"contextualized_question": "This is Question 1"}

    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "Yes",
            "flow_id": "behaviour-pre-assessment",
            "question_number": 0,
            "previous_question": "intro message text",
            "failure_count": 0,
        },
    )
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "This is Question 1"
    assert json_response["next_question"] == 1


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_intro_response",
    return_value={
        "action": "ABORT",
        "message": "Flow aborted.",
        "intent": "USER_ABORT",
        "intent_related_response": None,
    },
)
def test_assessment_consent_aborts(mock_handle_intro):
    """
    Tests that after an intro, an 'ABORT' action from the task
    results in the API returning the abort message and ending the flow.
    """
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "No",
            "flow_id": "behaviour-pre-assessment",
            "question_number": 0,
            "previous_question": "intro message text",
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Flow aborted."
    assert json_response["next_question"] is None
    mock_handle_intro.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_question")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_intro_response",
    return_value={
        "action": "REPROMPT",
        "message": "Re-prompt message",
        "intent": "REPAIR",
        "intent_related_response": None,
    },
)
async def test_repair_on_intro_consent(mock_handle_intro, mock_get_q):
    """
    Tests that a repair is triggered if the user gives a confusing response
    to the initial consent question.
    """
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "maybe?",
            "flow_id": "behaviour-pre-assessment",
            "question_number": 0,
            "previous_question": "Intro message",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "Re-prompt message"
    assert json_response["next_question"] == 0
    assert json_response["failure_count"] == 0
    mock_handle_intro.assert_called_once()
    mock_get_q.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow_id, question_number, endpoint",
    [
        ("dma-pre-assessment", 1, "/v1/assessment"),
        ("onboarding", 1, "/v1/onboarding"),
        ("anc-survey", "start", "/v1/survey"),
    ],
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
    return_value=[ChatMessage.from_assistant("Original question?")],
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_conversational_repair",
    return_value="This is the rephrased question.",
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.extract_anc_data_from_response",
    side_effect=lambda user_response,
    user_context,
    step_title,
    contextualized_question: (user_context, None),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.process_onboarding_step", return_value=({}, None)
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.validate_assessment_answer",
    return_value={"processed_user_response": None},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
async def test_flow_repair_on_invalid_answer(
    mock_handle_message,
    mock_validate,
    mock_process_onboarding,
    mock_extract_anc,
    mock_repair,
    mock_get_history,
    flow_id,
    question_number,
    endpoint,
):
    """
    Tests that if answer validation fails, the conversational repair is triggered across all flows.
    """
    if endpoint == "/v1/onboarding":
        json_payload = {
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "invalid answer",
            "failure_count": 0,
        }
    elif endpoint == "/v1/survey":
        json_payload = {
            "user_id": "TestUser",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "invalid answer",
            "failure_count": 0,
        }
    else:
        json_payload = {
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "invalid answer",
            "flow_id": flow_id,
            "question_number": question_number,
            "previous_question": "Original question?",
            "failure_count": 0,
        }

    client = TestClient(app)
    response = client.post(
        endpoint, headers={"Authorization": "Token testtoken"}, json=json_payload
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["question"] == "This is the rephrased question."
    assert json_response["intent"] == "REPAIR"
    assert json_response["failure_count"] == 1
    mock_repair.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow_id, question_number, endpoint",
    [
        ("dma-pre-assessment", 1, "/v1/assessment"),
        ("onboarding", 1, "/v1/onboarding"),
        ("anc-survey", "start", "/v1/survey"),
    ],
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
    return_value=[ChatMessage.from_assistant("Original question?")],
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_anc_survey_question",
    new_callable=mock.AsyncMock,
    return_value={"contextualized_question": "Next survey Q"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={"contextualized_question": "Next onboarding Q"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question",
    new_callable=mock.AsyncMock,
    return_value={"contextualized_question": "Next assessment Q"},
)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch(
    "ai4gd_momconnect_haystack.api.extract_anc_data_from_response",
    side_effect=lambda user_response,
    user_context,
    step_title,
    contextualized_question: (user_context, None),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.process_onboarding_step", return_value=({}, None)
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.validate_assessment_answer",
    return_value={"processed_user_response": None},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
async def test_flow_repair_escape_hatch(
    mock_handle_message,
    mock_validate,
    mock_process_onboarding,
    mock_extract_anc,
    mock_repair,
    mock_get_assessment_q,
    mock_get_onboarding_q,
    mock_get_survey_q,
    mock_get_history,
    flow_id,
    question_number,
    endpoint,
):
    """
    Tests that if a user fails validation twice, they are force-skipped to the next question across all flows.
    """
    if endpoint == "/v1/onboarding":
        json_payload = {
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "another invalid",
            "failure_count": 1,
        }
    elif endpoint == "/v1/survey":
        json_payload = {
            "user_id": "TestUser",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "another invalid",
            "failure_count": 1,
        }
    else:
        json_payload = {
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "another invalid",
            "flow_id": flow_id,
            "question_number": question_number,
            "previous_question": "Original question?",
            "failure_count": 1,
        }

    client = TestClient(app)
    response = client.post(
        endpoint, headers={"Authorization": "Token testtoken"}, json=json_payload
    )

    assert response.status_code == 200
    json_response = response.json()

    if endpoint == "/v1/assessment":
        assert json_response["processed_answer"] == "Skip"
        assert json_response["next_question"] == 2

    assert "Next" in json_response["question"]
    assert json_response["failure_count"] == 0
    mock_repair.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={"contextualized_question": "What is your area type?"},
)
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("SKIP_QUESTION", ""),
)
@mock.patch("ai4gd_momconnect_haystack.api.all_onboarding_questions")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding_skip_question(
    mock_get_history,
    mock_all_questions,
    mock_handle_user_message,
    mock_process_step,
    mock_get_next_q,
    mock_save_history,
):
    """
    Tests that if a user wants to skip an onboarding question, the context is
    updated with "Skipped" and the flow proceeds to the next question without
    triggering a validation failure or repair.
    """
    initial_history = [
        ChatMessage.from_system("..."),
        ChatMessage.from_assistant("Intro..."),
        ChatMessage.from_user("Yes"),
        ChatMessage.from_assistant("What province do you live in?"),
    ]
    mock_get_history.return_value = initial_history
    mock_q1 = mock.Mock()
    mock_q1.question_number = 1
    mock_q1.collects = "province"
    mock_q1.content = "What province do you live in?"
    mock_all_questions.__iter__.return_value = [mock_q1]

    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {"province": None},
            "user_input": "skip",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    json_response = response.json()

    assert json_response["question"] == "What is your area type?"
    assert json_response["user_context"] == {"province": "Skip"}
    assert json_response["results_to_save"] == ["province"]
    mock_process_step.assert_not_called()
    mock_handle_user_message.assert_called_once()
    mock_get_next_q.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question",
    new_callable=mock.AsyncMock,
    return_value={"contextualized_question": "This is Question 2"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.calculate_and_store_assessment_result",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.score_assessment_question", return_value=0)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("SKIP_QUESTION", ""),
)
async def test_assessment_skip_question(
    mock_handle_user_message,
    mock_validate_answer,
    mock_save_q,
    mock_score_q,
    mock_calc_result,
    mock_get_q,
):
    """
    Tests that if a user wants to skip an assessment question, the answer is
    recorded as "Skip" and the flow proceeds to the next question without
    triggering validation or repair.
    """
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_context": {},
            "user_input": "I don't want to answer",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "This is Question 1",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    json_response = response.json()

    assert json_response["question"] == "This is Question 2"
    assert json_response["next_question"] == 2
    assert json_response["processed_answer"] == "Skip"
    mock_validate_answer.assert_not_called()
    mock_handle_user_message.assert_called_once()
    assert mock_save_q.await_count == 2
    call_to_save_answer = mock.call(
        user_id="TestUser",
        assessment_type=AssessmentType.dma_pre_assessment,
        question_number=1,
        question="This is Question 1",
        user_response="Skip",
        score=0,
    )
    mock_save_q.assert_has_awaits([call_to_save_answer], any_order=True)
    mock_get_q.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_summary_confirmation_step")
async def test_onboarding_api_summary_state_handles_update(mock_summary_handler):
    """
    Tests that the API correctly calls the summary handler task when in the
    'confirming_summary' state and returns its result.
    """
    # Arrange
    mock_summary_handler.return_value = {
        "question": "Thank you! I've updated your information.",
        "user_context": {"province": "Western Cape", "area_type": "City"},
        "intent": "ONBOARDING_UPDATE_COMPLETE",
        "results_to_save": ["province"],
    }

    client = TestClient(app)
    # Act
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user-123",
            "user_input": "change province to western cape",
            "user_context": {"flow_state": "confirming_summary", "province": "Gauteng"},
        },
    )

    # Assert
    assert response.status_code == 200
    mock_summary_handler.assert_called_once()
    response_data = response.json()
    assert response_data["question"] == "Thank you! I've updated your information."
    assert response_data["intent"] == "ONBOARDING_UPDATE_COMPLETE"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding_api_flow_transitions_to_summary(
    mock_get_history, mock_handle_message, mock_process_step
):
    """
    Tests the full onboarding conversation flow up to the point where
    the final answer is given and the API transitions to the summary state.
    """
    # Arrange
    mock_get_history.return_value = [ChatMessage.from_assistant("Last question?")]
    mock_handle_message.return_value = ("JOURNEY_RESPONSE", "")
    mock_process_step.return_value = (
        {"province": "Gauteng", "area_type": "City"},
        None,  # NO more questions
    )

    client = TestClient(app)
    # Act
    final_response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user-flow",
            "user_input": "City",
            "user_context": {"province": "Gauteng"},
        },
    )

    # Assert
    assert final_response.status_code == 200
    response_data = final_response.json()

    assert response_data["intent"] == "AWAITING_SUMMARY_CONFIRMATION"
    assert "Here's the information I have for you:" in response_data["question"]
    assert "_*Province*: Gauteng_" in response_data["question"]
    assert response_data["user_context"]["flow_state"] == "confirming_summary"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch("ai4gd_momconnect_haystack.api.handle_intro_response")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_onboarding_consent_proceeds_directly_to_first_question(
    mock_get_history, mock_handle_intro, mock_process_step, mock_get_next_q
):
    """
    BUG FIX VERIFICATION:
    Tests that after a user gives consent, the API immediately gets the first
    real question and does NOT re-process the consent word ("yes") as a journey response.
    This prevents the incorrect "conversation repair" from being triggered.
    """
    # Arrange
    # Simulate the chat history right before the user gives consent
    mock_get_history.return_value = [
        ChatMessage.from_system("..."),
        ChatMessage.from_assistant("Shall we begin? 😊"),
    ]
    # Simulate a successful consent action
    mock_handle_intro.return_value = {"action": "PROCEED"}
    # Mock the function that fetches the next question
    mock_get_next_q.return_value = {
        "contextualized_question": "This is the first real question."
    }

    client = TestClient(app)

    # Act: The user sends a positive consent
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-consent-bug",
            "user_input": "yes",
            "user_context": {},
        },
    )

    # Assert
    assert response.status_code == 200
    json_response = response.json()

    # 1. The user should receive the first real question, not a repair message.
    assert json_response["question"] == "This is the first real question."
    assert json_response["intent"] != "REPAIR"

    # 2. Crucially, the main Q&A processor should NOT have been called with the consent word.
    mock_process_step.assert_not_called()

    # 3. The function to get the first question SHOULD have been called.
    mock_get_next_q.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_summary_confirmation_step")
async def test_onboarding_summary_confirmation_signals_start_dma(mock_summary_handler):
    """
    BUG FIX VERIFICATION:
    Tests that after the user confirms their data on the summary screen, the API
    response includes the specific 'ONBOARDING_COMPLETE_START_DMA' intent,
    signaling the caller to start the next flow.
    """
    # Arrange: Mock the summary task to return the specific DMA intent
    mock_summary_handler.return_value = {
        "question": "Perfect, thank you! Now for the next section.",
        "user_context": {"province": "Gauteng", "flow_state": None},
        "intent": "ONBOARDING_COMPLETE_START_DMA",
        "results_to_save": [],
    }

    client = TestClient(app)
    # Act: The user confirms their data
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-dma-bug",
            "user_input": "yes",
            "user_context": {"flow_state": "confirming_summary", "province": "Gauteng"},
        },
    )

    # Assert
    assert response.status_code == 200
    json_response = response.json()

    # The intent MUST be the specific signal for the caller to start the DMA assessment.
    assert json_response["intent"] == "ONBOARDING_COMPLETE_START_DMA"
    assert json_response["question"] == "Perfect, thank you! Now for the next section."


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_resume_endpoint_user_not_found(client: TestClient):
    """
    Tests that the /v1/resume endpoint returns a 404 if no journey state is found.
    """
    response = client.post(
        "/v1/resume",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "nonexistent-user"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_resume_endpoint_success(client: TestClient):
    """
    Tests that the /v1/resume endpoint successfully returns the flow_id for a user with a saved state.
    """
    # Arrange: Manually save a state for a user directly to the test DB
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(
                UserJourneyState(
                    user_id="resumable-user",
                    current_flow_id="onboarding",
                    current_step_identifier="2",
                    last_question_sent="What is your area type?",
                    user_context={"province": "Gauteng"},
                )
            )
            await session.commit()

    # Act
    response = client.post(
        "/v1/resume",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "resumable-user"},
    )

    # Assert
    assert response.status_code == 200
    assert response.json() == {"resume_flow_id": "onboarding"}


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={
        "contextualized_question": "What kind of area do you live in?",
        "question_number": 2,
    },
)
async def test_onboarding_full_resumption_flow(mock_get_next_q, client: TestClient):
    """
    Tests the full resumption flow: API receives resume flag, fetches saved context,
    and returns the correct resume message prompt.
    """
    # Arrange: Save a state for a user who has answered the first question
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(
                UserJourneyState(
                    user_id="onboarding-resume-user",
                    current_flow_id="onboarding",
                    current_step_identifier="1",
                    last_question_sent="Which province do you live in?",
                    user_context={"province": "Gauteng", "reminder_count": 0},
                )
            )
            await session.commit()

    # Act
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "onboarding-resume-user",
            "user_input": "",
            "user_context": {"resume": True},
            "failure_count": 0,
        },
    )

    # Assert
    assert response.status_code == 200
    json_response = response.json()

    # UPDATED ASSERTION: The first resumption call should now return the resume message,
    # not the next question directly.
    assert "Hi! Ready to pick up where you left off" in json_response["question"]
    assert json_response["intent"] == "SYSTEM_REMINDER_PROMPT"

    # The get_next_onboarding_question function should NOT be called yet.
    mock_get_next_q.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("REQUEST_TO_BE_REMINDED", None),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
    return_value=[ChatMessage.from_assistant("Some question")],
)
async def test_reminder_request_returns_reengagement_info(
    mock_get_history, mock_handle_message, client: TestClient
):
    """
    Tests that when a user asks for a reminder, the API response includes the reengagement_info object.
    """
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "reminder-user",
            "user_input": "remind me tomorrow",
            "user_context": {},
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "Some question",
            "failure_count": 0,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert "reengagement_info" in json_response
    assert json_response["reengagement_info"]["type"] == "USER_REQUESTED"
    assert "trigger_at_utc" in json_response["reengagement_info"]


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_reminder_response",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_user_journey_state", new_callable=mock.AsyncMock
)
async def test_survey_api_handles_reminder_response(
    mock_get_state, mock_handle_response, client: TestClient
):
    """
    Tests that the /v1/survey endpoint correctly routes a user who is responding
    to a reminder prompt to the new reminder handling logic.
    """
    # Arrange: Mock the user's state to indicate they are responding to a reminder
    mock_state = UserJourneyState(current_step_identifier="awaiting_reminder_response")
    mock_get_state.return_value = mock_state

    # Mock the response from the handler we are testing
    mock_handle_response.return_value = SurveyResponse(
        question="This was the original question.",
        user_context={},
        survey_complete=False,
        intent="JOURNEY_RESUMED",
        intent_related_response=None,
        results_to_save=[],
        failure_count=0,
    )

    # Act
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user",
            "survey_id": "anc",
            "user_input": "Yes",
            "user_context": {},
        },
    )

    # Assert
    assert response.status_code == 200
    assert response.json()["intent"] == "JOURNEY_RESUMED"
    assert response.json()["question"] == "This was the original question."

    # Verify that the correct logic path was taken
    mock_get_state.assert_awaited_once_with("test-user")
    mock_handle_response.assert_awaited_once_with("test-user", "Yes", mock_state)


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_user_journey_state", new_callable=mock.AsyncMock
)
async def test_api_re_engagement_flow_affirmative(mock_get_state, client: TestClient):
    """
    Tests the full API flow when a user says "Yes" to a re-engagement prompt.
    """
    # Arrange: Mock the user's state to be awaiting a reminder response
    mock_state = UserJourneyState(
        user_id="test-user",
        current_flow_id="anc-survey",
        current_step_identifier="awaiting_reminder_response",
        last_question_sent="This was the original question.",
        user_context={"reminder_count": 1},
    )
    mock_get_state.return_value = mock_state

    # Act: The user replies "Yes"
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user",
            "survey_id": "anc",
            "user_input": "a. Yes",
            "user_context": {"reminder_count": 1},
        },
    )

    # Assert: The API should return the original question
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["intent"] == "JOURNEY_RESUMED"
    assert json_response["question"] == "This was the original question."


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.tasks.handle_reminder_request",
    new_callable=mock.AsyncMock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_user_journey_state", new_callable=mock.AsyncMock
)
async def test_api_re_engagement_flow_remind_later(
    mock_get_state, mock_reminder_request, client: TestClient
):
    """
    Tests the full API flow when a user asks for another reminder.
    """
    # Arrange: Mock the user's state
    mock_state = UserJourneyState(
        user_id="test-user",
        current_flow_id="anc-survey",
        current_step_identifier="awaiting_reminder_response",
        last_question_sent="Original question.",
        user_context={"reminder_count": 1},
    )
    mock_get_state.return_value = mock_state

    # Arrange: Mock the return of the dependency that sets the next reminder
    mock_reminder_request.return_value = (
        "OK, we'll remind you again.",
        mock.Mock(spec=ReengagementInfo),
    )

    # Act: The user replies "Remind me tomorrow"
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user",
            "survey_id": "anc",
            "user_input": "b. Remind me tomorrow",
            "user_context": {"reminder_count": 1},
        },
    )

    # Assert: The API should return the acknowledgement message
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["intent"] == "REQUEST_TO_BE_REMINDED"
    assert json_response["question"] == "OK, we'll remind you again."
    assert "reengagement_info" in json_response


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_anc_survey_question", new_callable=mock.AsyncMock
)
@mock.patch("ai4gd_momconnect_haystack.api.extract_anc_data_from_response")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.AsyncMock,
)
async def test_api_anc_survey_full_turn_with_key_refactor(
    mock_get_history,
    mock_handle_message,
    mock_extract_data,
    mock_get_question,
    client: TestClient,
):
    """
    Tests a full, successful turn of the ANC survey to confirm the key-based
    refactor works end-to-end through the API.
    """
    # Arrange
    # 1. User has already passed the intro and received the first question ("start")
    initial_history = [
        ChatMessage.from_system("..."),
        ChatMessage.from_assistant("Intro", meta={"step_title": "intro"}),
        ChatMessage.from_user("Yes"),
        ChatMessage.from_assistant("Hi mom...", meta={"step_title": "start"}),
    ]
    mock_get_history.return_value = initial_history
    mock_handle_message.return_value = ("JOURNEY_RESPONSE", None)

    # 2. Mock the data extraction to return the NEW standardized key
    updated_context_with_key = {"start": "VISIT_YES"}
    mock_extract_data.return_value = (updated_context_with_key, None)

    # 3. Mock the final get_question call
    mock_get_question.return_value = {
        "contextualized_question": "Did a health worker see you?",
        "question_identifier": "Q_seen",
        "is_final_step": False,
    }

    # Act: User sends their response
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-user",
            "survey_id": "anc",
            "user_context": {},
            "user_input": "I went to the clinic",
            "failure_count": 0,
        },
    )

    # Assert
    assert response.status_code == 200
    assert response.json()["question"] == "Did a health worker see you?"

    # 4. CRITICAL: Verify that get_anc_survey_question was called with the
    #    context containing the standardized key. This proves the entire
    #    refactor is working.
    mock_get_question.assert_awaited_once_with(
        user_id="test-user",
        user_context=updated_context_with_key,
        chat_history=mock.ANY,
    )


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_resumption_from_awaiting_reminder_state_is_safe(client: TestClient):
    """
    BUG FIX VERIFICATION:
    Tests that if a user drops out *after* receiving a resume prompt,
    the system can safely re-engage them a second time without crashing.
    This specifically targets the `int('awaiting_reminder_response')` ValueError.
    """
    user_id = "double-dropout-user"

    # Arrange: Manually create the problematic state in the database.
    # The user was on step 5 of an assessment, got re-engaged once, and we are
    # now about to re-engage them a second time.
    async with AsyncSessionLocal() as session:
        async with session.begin():
            session.add(
                UserJourneyState(
                    user_id=user_id,
                    current_flow_id="dma-pre-assessment",
                    # This is the state that caused the crash
                    current_step_identifier="awaiting_reminder_response",
                    last_question_sent="Hi! Ready to pick up where you left off...?",
                    user_context={"reminder_count": 1},
                )
            )
            await session.commit()

    # Act: The platform re-engages the user a second time.
    response = client.post(
        "/v1/assessment",  # Calling the assessment endpoint to resume
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": user_id,
            "user_input": "",
            "user_context": {"resume": True},
            "flow_id": "dma-pre-assessment",  # This is needed for the assessment endpoint
            "question_number": 0,  # Not relevant here, but required by the model
            "previous_question": "",  # Not relevant here, but required by the model
        },
    )

    # Assert:
    # 1. The API call must succeed and not crash.
    assert response.status_code == 200
    json_response = response.json()

    # 2. The user should receive the resume prompt again.
    assert "Hi! Ready to pick up" in json_response["question"]

    # 3. CRITICAL: The `next_question` field must be None, because the step
    #    identifier was not a number. This proves the fix is working.
    assert json_response["next_question"] is None


@pytest.mark.parametrize(
    "scenario, user_id, mock_state, expected_state_calls",
    [
        ( "COMPLETED_SURVEY", "completed_user", 
          UserJourneyState(current_flow_id="anc-survey", current_step_identifier=""), 1),
        # For the NO_STATE case, we now correctly expect 2 calls
        ( "NO_STATE", "new_user", None, 2),
        ( "DIFFERENT_FLOW", "onboarding_user", 
          UserJourneyState(current_flow_id="onboarding", current_step_identifier="step_2"), 1),
    ],
    ids=["When survey is completed", "When no state exists", "When state is for another flow"],
)
@patch("ai4gd_momconnect_haystack.api.save_user_journey_state", new_callable=AsyncMock)
@patch("ai4gd_momconnect_haystack.api.get_anc_survey_question")
@patch("ai4gd_momconnect_haystack.api.handle_journey_resumption_prompt")
@patch("ai4gd_momconnect_haystack.api.get_user_journey_state")
async def test_survey_resume_is_bypassed_for_non_resumable_states(
    mock_get_state: AsyncMock,
    mock_handle_resume: AsyncMock,
    mock_get_question: AsyncMock,
    mock_save_state: AsyncMock,
    scenario: str,
    user_id: str,
    mock_state: UserJourneyState | None,
    expected_state_calls: int,
):
    """
    GIVEN a user's state is not a resumable survey (it's completed, null, or for another flow)
    WHEN they start the survey with a `resume: True` flag
    THEN the resume logic should be bypassed and a fresh survey should start.
    """
    # --- Arrange ---
    mock_get_state.return_value = mock_state
    mock_get_question.return_value = {
        "contextualized_question": "Welcome! Here is the first question.",
        "question_identifier": "start",
        "is_final_step": False,
    }
    
    request = SurveyRequest(
        user_id=user_id,
        survey_id=HistoryType.anc,
        user_input="",
        user_context={"resume": True},
    )

    # --- Act ---
    response = await survey(request, token="fake_token")

    # --- Assert ---
    # This assertion is now flexible and checks for the correct call count in each case
    assert mock_get_state.call_count == expected_state_calls
    
    mock_handle_resume.assert_not_called()
    mock_get_question.assert_called_once()
    assert response.question == "Welcome! Here is the first question."

@patch("ai4gd_momconnect_haystack.api.save_user_journey_state", new_callable=AsyncMock)
@patch("ai4gd_momconnect_haystack.api.get_anc_survey_question")
@patch("ai4gd_momconnect_haystack.api.handle_journey_resumption_prompt")
@patch("ai4gd_momconnect_haystack.api.get_user_journey_state")
async def test_survey_resume_is_triggered_for_in_progress_state(
    mock_get_state: AsyncMock,
    mock_handle_resume: AsyncMock,
    mock_get_question: AsyncMock,
    mock_save_state: AsyncMock,
):
    """
    GIVEN a user with an IN-PROGRESS survey state (step is a non-empty string)
    WHEN they start the survey with a `resume: True` flag
    THEN the resume logic should be triggered and the resumption handler called.
    """
    # --- Arrange ---
    mock_get_state.return_value = UserJourneyState(
        current_flow_id="anc-survey", current_step_identifier="Q_bp_measurement"
    )
    mock_handle_resume.return_value = {
        "question": "Welcome back! Let's continue.",
        "question_identifier": "Q_bp_measurement",
    }
    request = SurveyRequest(
        user_id="paused_user",
        survey_id=HistoryType.anc,
        user_input="",
        user_context={"resume": True},
    )

    # --- Act ---
    response = await survey(request, token="fake_token")

    # --- Assert ---
    mock_get_state.assert_called_once_with("paused_user")
    mock_handle_resume.assert_called_once_with(
        user_id="paused_user", flow_id="anc-survey"
    )
    mock_get_question.assert_not_called()
    assert response["question"] == "Welcome back! Let's continue."
