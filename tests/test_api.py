import os
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from haystack.dataclasses import ChatMessage
from sentry_sdk import get_client as get_sentry_client

from ai4gd_momconnect_haystack.api import app, setup_sentry
from ai4gd_momconnect_haystack.database import SessionLocal
from ai4gd_momconnect_haystack.enums import (
    AssessmentType,
    ExtractionStatus,
    HistoryType,
)
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndScoreBasedMessage,
    AssessmentResult,
    LegacySurveyResponse,
    OrchestratorSurveyRequest,
    ReengagementInfo,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    UserJourneyState,
)

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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding_chitchat(mock_get_history, mock_handle_user_message, mock_repair):
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA_TEXT", "Test Persona")
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_chat_history_for_user",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding_first_question(
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
    get_or_create_chat_history.assert_called_once_with(
        user_id="TestUser", history_type=HistoryType.onboarding
    )

    # 3. Assert that the old history was then deleted
    delete_chat_history_for_user.assert_called_once_with(
        "TestUser", HistoryType.onboarding
    )

    # 4. Assert that the new history is saved correctly
    save_chat_history.assert_called_once_with(
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding(
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
        "city",
        ExtractionStatus.SUCCESS,
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
def test_assessment_chitchat(mock_handle_user_message, mock_repair):
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_assessment_history_for_user",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.Mock,
)
def test_assessment_initial_message(
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
    delete_assessment_history_for_user.assert_called_once_with(
        "TestUser", AssessmentType.dma_pre_assessment
    )
    get_assessment_question.assert_called_once()
    save_assessment_question.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.calculate_and_store_assessment_result",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch("ai4gd_momconnect_haystack.api.score_assessment_question")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.Mock,
)
def test_assessment_valid_journey_response(
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

    calculate_and_store_assessment_result.assert_called_once_with(
        "TestUser", "dma-pre-assessment"
    )

    get_assessment_question.assert_called_once_with(
        user_id="TestUser",
        flow_id="dma-pre-assessment",
        question_number=2,  # Importantly, it asks for the *next* question
        user_context={},
    )

    # Assert that the data was saved correctly in two separate calls
    assert save_assessment_question.call_count == 2

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

    save_assessment_question.assert_has_calls(
        [call_to_save_answer, call_to_save_question], any_order=False
    )


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.Mock,
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
def test_assessment_end_initial_message(
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

    mock_get_result.assert_called_once()
    mock_get_history.assert_called_once()
    mock_save_message.assert_called_once_with("TestUser", "dma-pre-assessment", 1, "")


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch(
    "ai4gd_momconnect_haystack.api.assessment_end_flow_map",
    {
        "knowledge-pre-assessment": [
            AssessmentEndScoreBasedMessage.model_validate(
                {
                    "message_nr": 1,
                    "high-score-content": {"content": "KAB High score content"},
                    "medium-score-content": {"content": "KAB Medium score content"},
                    "low-score-content": {"content": "KAB Low score content"},
                    "skipped-many-content": {"content": "KAB Skipped many content"},
                }
            )
        ]
    },
)
def test_assessment_end_initial_message_kab_pre(
    mock_get_content,
    mock_save_message,
    mock_get_history,
    mock_get_result,
):
    """
    Tests the initial call to the endpoint with no user_input for a KAB pre assessment.
    It should return the first message of the assessment-end flow.
    """
    # --- Mock Setup ---
    mock_get_result.return_value = AssessmentResult(
        score=100.0, category="high", crossed_skip_threshold=False
    )
    mock_get_history.return_value = []
    mock_get_content.return_value = ("KAB High score content", None)

    # --- API Call ---
    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_input": "",
            "flow_id": "knowledge-pre-assessment",
        },
    )

    # --- Assertions ---
    assert response.status_code == 200
    assert response.json() == {
        "message": "KAB High score content",
        "task": "",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "reengagement_info": None,
    }

    mock_get_result.assert_called_once()
    mock_get_history.assert_called_once()
    mock_save_message.assert_called_once_with(
        "TestUser", "knowledge-pre-assessment", 1, ""
    )


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
def test_assessment_end_initial_message_kab_post(
    mock_get_content,
    mock_save_message,
    mock_get_history,
    mock_get_result,
):
    """
    Tests the initial call to the endpoint with no user_input for a KAB post assessment.
    It should return the first message of the assessment-end flow.
    """
    mock_get_result.return_value = AssessmentResult(
        score=100.0, category="high", crossed_skip_threshold=False
    )
    mock_get_history.return_value = []
    mock_get_content.return_value = ("KAB High score content", None)

    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "TestUser",
            "user_input": "",
            "flow_id": "knowledge-post-assessment",
        },
    )

    # Desired behavior once implemented
    assert response.status_code == 200
    assert response.json() == {
        "message": "KAB High score content",
        "task": "",
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "reengagement_info": None,
    }


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_end_response")
@mock.patch("ai4gd_momconnect_haystack.api.determine_task")
@mock.patch("ai4gd_momconnect_haystack.api.assessment_end_flow_map")
@mock.patch("ai4gd_momconnect_haystack.api.response_is_required_for")
def test_assessment_end_valid_response_to_required_question(
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
    assert mock_save_message.call_count == 2
    call_to_save_answer = mock.call("TestUser", "dma-pre-assessment", 2, "Yes")
    call_to_save_question = mock.call("TestUser", "dma-pre-assessment", 3, "")
    mock_save_message.assert_has_calls(
        [call_to_save_answer, call_to_save_question], any_order=True
    )


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_end_message",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_end_response")
@mock.patch("ai4gd_momconnect_haystack.api.assessment_end_flow_map")
@mock.patch("ai4gd_momconnect_haystack.api.response_is_required_for")
def test_assessment_end_invalid_response(
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
    mock_save_message.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_result")
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history")
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.handle_reminder_request")
def test_assessment_end_remind_me_later_triggers_reminder(
    mock_handle_reminder,
    mock_handle_user_message,
    mock_get_content,
    mock_get_history,
    mock_get_result,
):
    """
    Tests that when handle_user_message returns REQUEST_TO_BE_REMINDED the reminder
    flow is triggered (handle_reminder_request) and its response is returned.
    """
    # --- Mock Setup ---
    mock_get_result.return_value = AssessmentResult(
        score=50.0, category="medium", crossed_skip_threshold=False
    )
    # Simulate that the last saved message was message number 2
    mock_get_history.return_value = [
        AssessmentEndMessagingHistory(message_number=2, user_response="")
    ]

    mock_get_content.side_effect = [
        ("Would you like a reminder? (Yes/No)", ["Yes", "No"]),
        ("Okay, we'll handle that.", None),
    ]

    mock_handle_user_message.return_value = ("REQUEST_TO_BE_REMINDED", "")

    # Mock the reminder handler to return a message and a valid ReengagementInfo
    mock_handle_reminder.return_value = (
        "We'll remind you",
        {
            "type": "REMINDER",
            "trigger_at_utc": datetime.now(timezone.utc),
            "flow_id": "dma-pre-assessment",
            "reminder_type": 2,
        },
    )

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
    json_resp = response.json()
    assert json_resp["message"] == "We'll remind you"
    assert json_resp["task"] == ""
    assert json_resp["intent"] == "REQUEST_TO_BE_REMINDED"
    assert json_resp["intent_related_response"] == ""
    # ReengagementInfo fields should be present and include our flow_id and reminder_type
    assert json_resp["reengagement_info"]["flow_id"] == "dma-pre-assessment"
    assert json_resp["reengagement_info"]["reminder_type"] == 2

    mock_handle_reminder.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_assessment_end_resume_flag():
    """
    Tests that when the request contains a resume flag, the assessment-end
    endpoint uses the saved `UserJourneyState` to build the resume response.
    This inserts a real `UserJourneyState` into the test DB instead of mocking
    the resumption handler.
    """
    # Arrange: persist a state for this user in the test DB
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="resume-user",
                    current_flow_id="dma-pre-assessment",
                    current_step_identifier="1",
                    last_question_sent="Previous question?",
                    user_context={"reminder_count": 0},
                )
            )
            session.commit()

    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "resume-user",
            "user_input": "",
            "flow_id": "dma-pre-assessment",
            "user_context": {"resume": True},
        },
    )

    assert response.status_code == 200
    json_resp = response.json()
    # We should receive the resume meta-prompt message
    assert isinstance(json_resp, dict)
    assert "message" in json_resp or "question" in json_resp
    # The resume message should include an invitation to continue
    msg_text = json_resp.get("message") or json_resp.get("question")
    assert (
        "Ready to pick up" in msg_text
        or "Ready to pick up where you left off" in msg_text
    )


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
def test_assessment_end_response_to_awaiting_reminder(mock_get_result):
    """
    Tests that when the user's saved state indicates they are awaiting a reminder
    response, the assessment-end endpoint calls `handle_reminder_response` and
    returns its result.
    """

    # Arrange: persist a state record indicating the user is awaiting a reminder response
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="reminder-user",
                    current_flow_id="dma-pre-assessment",
                    current_step_identifier="awaiting_reminder_response",
                    last_question_sent="Please reply",
                    user_context={"reminder_count": 1},
                )
            )
            session.commit()

    mock_get_result.return_value = AssessmentResult(
        score=50.0, category="medium", crossed_skip_threshold=False
    )

    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "reminder-user",
            "user_input": "Yes",
            "flow_id": "dma-pre-assessment",
        },
    )

    assert response.status_code == 200
    json_resp = response.json()
    # For an assessment flow we expect the journey to be resumed
    assert json_resp["intent"] == "JOURNEY_RESUMED"
    # The original last question should be re-sent
    assert json_resp["message"] == "Please reply"


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
def test_assessment_end_response_to_awaiting_reminder_request_reminder(mock_get_result):
    """
    Tests that when the user's saved state indicates they are awaiting a reminder
    response, the assessment-end endpoint calls `handle_reminder_response` and
    returns its result. If they request a reminder again, the correct intent is returned.
    """

    # Arrange: persist a state record indicating the user is awaiting a reminder response
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="reminder-user",
                    current_flow_id="dma-pre-assessment",
                    current_step_identifier="awaiting_reminder_response",
                    last_question_sent="Please reply",
                    user_context={"reminder_count": 1},
                )
            )
            session.commit()

    mock_get_result.return_value = AssessmentResult(
        score=50.0, category="medium", crossed_skip_threshold=False
    )

    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "reminder-user",
            "user_input": "Remind me",
            "flow_id": "dma-pre-assessment",
        },
    )

    assert response.status_code == 200
    json_resp = response.json()
    # For an assessment flow we expect the journey to be resumed
    assert json_resp["intent"] == "REQUEST_TO_BE_REMINDED"
    # The original last question should be re-sent
    assert "Great! We’ll remind you tomorrow" in json_resp["message"]


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_result", new_callable=mock.Mock
)
def test_assessment_end_response_to_awaiting_reminder_ambiguous(mock_get_result):
    """
    Tests that when the user's saved state indicates they are awaiting a reminder
    response, the assessment-end endpoint calls `handle_reminder_response` and
    returns its result. If they send something random respond accordingly
    """

    # Arrange: persist a state record indicating the user is awaiting a reminder response
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="ambiguous-user",
                    current_flow_id="dma-pre-assessment",
                    current_step_identifier="awaiting_reminder_response",
                    last_question_sent="Please reply",
                    user_context={"reminder_count": 1},
                )
            )
            session.commit()

    mock_get_result.return_value = AssessmentResult(
        score=50.0, category="medium", crossed_skip_threshold=False
    )

    client = TestClient(app)
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "ambiguous-user",
            "user_input": "What is this",
            "flow_id": "dma-pre-assessment",
        },
    )

    assert response.status_code == 200
    json_resp = response.json()
    # For an assessment flow we expect the journey to be resumed
    assert json_resp["intent"] == "REPAIR"
    # The original last question should be re-sent
    assert "Hi! Ready to pick up where you left off?" in json_resp["message"]


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
    get_sentry_client().close()


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
    new_callable=mock.Mock,
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
    mock_delete_history.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.delete_assessment_history_for_user",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.Mock
)
def test_assessment_initial_message_skips_intro(
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
    mock_delete_history.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.Mock
)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.handle_intro_response")
def test_assessment_consent_proceeds(
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
        "message": "Hello! Shall we begin?",
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
def test_repair_on_intro_consent(mock_handle_intro, mock_get_q):
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


@pytest.mark.parametrize(
    "flow_id, question_number, endpoint",
    [
        ("dma-pre-assessment", 1, "/v1/assessment"),
        ("onboarding", 1, "/v1/onboarding"),
    ],
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
    return_value=[ChatMessage.from_assistant("Original question?")],
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_conversational_repair",
    return_value="This is the rephrased question.",
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.process_onboarding_step",
    return_value=({}, None, "invalid answer", ExtractionStatus.NO_MATCH),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.validate_assessment_answer",
    return_value={"processed_user_response": None},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
def test_flow_repair_on_invalid_answer(
    mock_handle_message,
    mock_validate,
    mock_process_onboarding,
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


@pytest.mark.parametrize(
    "flow_id, question_number, endpoint",
    [
        ("dma-pre-assessment", 1, "/v1/assessment"),
        ("onboarding", 1, "/v1/onboarding"),
    ],
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
    return_value=[ChatMessage.from_assistant("Original question?")],
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={"contextualized_question": "Next onboarding Q"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question",
    new_callable=mock.Mock,
    return_value={"contextualized_question": "Next assessment Q"},
)
@mock.patch("ai4gd_momconnect_haystack.api.handle_conversational_repair")
@mock.patch(
    "ai4gd_momconnect_haystack.api.process_onboarding_step",
    return_value=({}, None, "another invalid", ExtractionStatus.NO_MATCH),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.validate_assessment_answer",
    return_value={"processed_user_response": None},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("JOURNEY_RESPONSE", ""),
)
def test_flow_repair_escape_hatch(
    mock_handle_message,
    mock_validate,
    mock_process_onboarding,
    mock_repair,
    mock_get_assessment_q,
    mock_get_onboarding_q,
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_chat_history",
    new_callable=mock.Mock,
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={"contextualized_question": "What is your area type?"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("SKIP_QUESTION", ""),
)
@mock.patch("ai4gd_momconnect_haystack.api.all_onboarding_questions")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding_skip_question(
    mock_get_history,
    mock_all_questions,
    mock_handle_user_message,
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

    mock_handle_user_message.assert_called_once()
    mock_get_next_q.assert_called_once()
    mock_save_history.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question",
    new_callable=mock.Mock,
    return_value={"contextualized_question": "This is Question 2"},
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.calculate_and_store_assessment_result",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.score_assessment_question", return_value=0)
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.Mock,
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("SKIP_QUESTION", ""),
)
def test_assessment_skip_question(
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
    assert mock_save_q.call_count == 2
    call_to_save_answer = mock.call(
        user_id="TestUser",
        assessment_type=AssessmentType.dma_pre_assessment,
        question_number=1,
        question="This is Question 1",
        user_response="Skip",
        score=0,
    )
    mock_save_q.assert_has_calls([call_to_save_answer], any_order=True)
    mock_get_q.assert_called_once()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_summary_confirmation_step")
def test_onboarding_api_summary_state_handles_update(mock_summary_handler):
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding_api_flow_transitions_to_summary(
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
        None,
        "City",
        ExtractionStatus.SUCCESS,
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.process_onboarding_step")
@mock.patch("ai4gd_momconnect_haystack.api.handle_intro_response")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
)
def test_onboarding_consent_proceeds_directly_to_first_question(
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_summary_confirmation_step")
def test_onboarding_summary_confirmation_signals_start_dma(mock_summary_handler):
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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_resume_endpoint_user_not_found(client: TestClient):
    """
    Tests that the /v1/resume endpoint returns a 404 if no journey state is found.
    """
    response = client.post(
        "/v1/resume",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "nonexistent-user"},
    )
    assert response.status_code == 404


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_resume_endpoint_success(client: TestClient):
    """
    Tests that the /v1/resume endpoint successfully returns the flow_id for a user with a saved state.
    """
    # Arrange: Manually save a state for a user directly to the test DB
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="resumable-user",
                    current_flow_id="onboarding",
                    current_step_identifier="2",
                    last_question_sent="What is your area type?",
                    user_context={"province": "Gauteng"},
                )
            )
            session.commit()

    # Act
    response = client.post(
        "/v1/resume",
        headers={"Authorization": "Token testtoken"},
        json={"user_id": "resumable-user"},
    )

    # Assert
    assert response.status_code == 200
    assert response.json() == {"resume_flow_id": "onboarding"}


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
    return_value={
        "contextualized_question": "What kind of area do you live in?",
        "question_number": 2,
    },
)
def test_onboarding_full_resumption_flow(mock_get_next_q, client: TestClient):
    """
    Tests the full resumption flow: API receives resume flag, fetches saved context,
    and returns the correct resume message prompt.
    """
    # Arrange: Save a state for a user who has answered the first question
    with SessionLocal() as session:
        with session.begin():
            session.add(
                UserJourneyState(
                    user_id="onboarding-resume-user",
                    current_flow_id="onboarding",
                    current_step_identifier="1",
                    last_question_sent="Which province do you live in?",
                    user_context={"province": "Gauteng", "reminder_count": 0},
                )
            )
            session.commit()

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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.handle_user_message",
    return_value=("REQUEST_TO_BE_REMINDED", None),
)
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
    new_callable=mock.Mock,
    return_value=[ChatMessage.from_assistant("Some question")],
)
def test_reminder_request_returns_reengagement_info(
    mock_get_history, mock_handle_message, client: TestClient
):
    """
    Tests that when a user asks for a reminder, the API response includes the reengagement_info object.
    """
    # Freeze 'now' used by the reminder logic to make the expected trigger time deterministic
    fixed_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    expected_trigger = fixed_now + timedelta(hours=23)

    # Patch the datetime used inside the tasks module so handle_reminder_request
    # computes trigger_at_utc from our fixed time.
    with mock.patch("ai4gd_momconnect_haystack.tasks.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now

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
    reeng = json_response["reengagement_info"]
    assert reeng["type"] == "USER_REQUESTED"
    # The assessment endpoint uses reminder_type=2 for user requested reminders
    assert reeng["flow_id"] == "dma-pre-assessment"
    assert reeng["reminder_type"] == 2
    # trigger_at_utc is serialized as an ISO string; parse and compare to expected
    parsed = datetime.fromisoformat(reeng["trigger_at_utc"])
    assert parsed == expected_trigger


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_resumption_from_awaiting_reminder_state_is_safe(client: TestClient):
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
    with SessionLocal() as session:
        with session.begin():
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
            session.commit()

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

    # 3. CRITICAL: The `next_question` field must be 0, we're using next_question_number
    # and it is not set yet if resuming from intro
    assert json_response["next_question"] == 0


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch(
    "ai4gd_momconnect_haystack.api.survey_orchestrator.process_survey_turn",
    new_callable=mock.Mock,
)
def test_survey_endpoint_passes_through_to_orchestrator(
    mock_process_turn, client: TestClient
):
    """
    Tests that the /v1/survey endpoint correctly forwards the request to the
    survey_orchestrator and returns its response.
    """
    # Arrange: Define the request payload that the API will receive
    request_payload = {
        "user_id": "test-orchestrator-user",
        "survey_id": "anc-survey",
        "user_input": "Yes, I did",
        "user_context": {"some_key": "some_value"},
        "failure_count": 0,
    }

    # Arrange: Define the response that the mocked orchestrator will return
    mock_orchestrator_response = LegacySurveyResponse(
        question="This is the next question from the orchestrator.",
        question_identifier="Q2",
        user_context={"some_key": "some_value", "new_key": "new_value"},
        survey_complete=False,
        intent="JOURNEY_RESPONSE",
        intent_related_response=None,
        results_to_save=["new_key"],
    )
    mock_process_turn.return_value = mock_orchestrator_response

    # Act: Call the /v1/survey API endpoint
    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json=request_payload,
    )

    # Assert
    # 1. Check that the API response is successful and matches the orchestrator's response
    assert response.status_code == 200
    assert response.json() == mock_orchestrator_response.model_dump()

    # 2. Verify that the orchestrator was called exactly once with the correct request object
    mock_process_turn.assert_called_once()
    call_args = mock_process_turn.call_args.args
    assert len(call_args) == 1
    assert isinstance(call_args[0], OrchestratorSurveyRequest)
    assert call_args[0].user_id == "test-orchestrator-user"
    assert call_args[0].user_input == "Yes, I did"


@pytest.mark.parametrize(
    "flow_id, user_input, processed_response, should_remind, expected_q1_content, expected_next_message",
    [
        # Reminder Paths (with full, correct expected content)
        (
            "knowledge-pre-assessment",
            "Remind me tomorrow",
            "REMIND_TOMORROW",
            True,
            "Great!\n\nRemember, you can skip any question - just type and send `skip`.\n\n*When do you think a pregnant woman should get her first pregnancy check-up?* 🤰🏽\n\na. Before 14 weeks\nb. At about 16 weeks\nc. At about 20 weeks\nd. At about 24 weeks\ne. At birth\nf. I don't know",
            None,
        ),
        (
            "dma-pre-assessment",
            "Yes",
            "Yes",
            True,
            "How much do you agree or disagree with this statement:\n\n*I feel like I can make decisions about my health.*\n\na. I strongly disagree 👎👎\nb. I disagree 👎\nc. I'm not sure\nd. I agree 👍\ne. I strongly agree 👍👍",
            None,
        ),
        # Proceed Paths (verify the exact next message)
        (
            "behaviour-pre-assessment",
            "Yes, let's go",
            "Yes",
            False,
            None,
            "",
        ),  # This flow ends, so the next message is empty.
        (
            "attitude-pre-assessment",
            "No",
            "No",
            False,
            None,
            "Thanks for your feedback. We'll be back with some follow-up questions in the next few weeks! 💕\n\nClick on this link to go to MomConnect: https://wa.me/27796312456?text=menu",
        ),
    ],
)
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_content_from_message_data")
@mock.patch("ai4gd_momconnect_haystack.api.response_is_required_for", return_value=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_reminder_request")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_end_response")
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_end_messaging_history")
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_result")
@mock.patch("ai4gd_momconnect_haystack.api.determine_task")
@mock.patch("ai4gd_momconnect_haystack.api.delete_assessment_history_for_user")
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
def test_assessment_end_skipped_many_scenarios_robust(
    mock_handle_user_message,
    mock_delete_history,
    mock_determine_task,
    mock_get_result,
    mock_get_history,
    mock_validate_response,
    mock_handle_reminder,
    mock_response_required,
    mock_get_content,
    client: TestClient,
    flow_id,
    user_input,
    processed_response,
    should_remind,
    expected_q1_content,
    expected_next_message,
):
    """
    Tests the two main outcomes for the 'skipped-many' scenario:
    1. The user requests a reminder, which should restart the assessment.
    2. The user proceeds, which should continue to the correct next message.
    """
    # --- MOCK SETUP ---
    mock_handle_user_message.return_value = ("JOURNEY_RESPONSE", "")
    mock_determine_task.return_value = ""
    mock_get_result.return_value = AssessmentResult(
        score=0, category="low", crossed_skip_threshold=True
    )
    mock_get_history.return_value = [
        AssessmentEndMessagingHistory(message_number=1, user_response="")
    ]
    mock_validate_response.return_value = {
        "processed_user_response": processed_response,
        "next_message_number": 2,
    }
    mock_handle_reminder.return_value = (
        "Reminder set!",
        ReengagementInfo(
            type="USER_REQUESTED",
            trigger_at_utc=datetime.now(timezone.utc),
            flow_id=flow_id,
            reminder_type=2,
        ),
    )

    # Make the mock for get_content_from_message_data intelligent.
    if flow_id == "attitude-pre-assessment" or flow_id == "dma-pre-assessment":
        valid_responses_for_msg1 = ["Yes", "No"]
    else:
        valid_responses_for_msg1 = ["Yes", "Remind me tomorrow"]

    mock_get_content.side_effect = [
        (
            "Prompt message for message 1",
            valid_responses_for_msg1,
        ),  # Return value for message 1
        (expected_next_message, []),  # Return value for message 2
    ]

    # --- API CALL ---
    response = client.post(
        "/v1/assessment-end",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_id": "test-skipped-many-user",
            "user_input": user_input,
            "flow_id": flow_id,
        },
    )

    # --- ASSERTIONS ---
    assert response.status_code == 200
    json_response = response.json()

    if should_remind:
        mock_delete_history.assert_called_once_with("test-skipped-many-user", flow_id)
        mock_handle_reminder.assert_called_once()
        call_args = mock_handle_reminder.call_args.kwargs
        assert call_args["step_identifier"] == "1"
        assert call_args["last_question"] == expected_q1_content
    else:
        mock_delete_history.assert_not_called()
        mock_handle_reminder.assert_not_called()
        assert json_response["message"] == expected_next_message
