import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from haystack.dataclasses import ChatMessage
from sentry_sdk import get_client as get_sentry_client

from ai4gd_momconnect_haystack.api import app, setup_sentry
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndScoreBasedMessage,
    AssessmentResult,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import AssessmentEndMessagingHistory

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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
def test_onboarding_invalid_auth_token():
    client = TestClient(app)
    response = client.post("/v1/onboarding", headers={"Authorization": "Token invalid"})
    assert response.status_code == 401


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_onboarding_chitchat():
    """
    If the user's response gets classified as chitchat, then we should not try
    to extract the onboarding data. Also, the response to the chitchat should be
    captured in the message history
    """
    initial_history = [ChatMessage.from_assistant(text="Hi!")]
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
            return_value=initial_history,
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message",
            return_value=("CHITCHAT", "User is chitchatting"),
        ) as _,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
            return_value={
                "contextualized_question": "Which province are you currently living in? 🏡"
            },
        ) as _,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/onboarding",
            headers={"Authorization": "Token testtoken"},
            json={"user_id": "TestUser", "user_context": {}, "user_input": "Hello!"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Which province are you currently living in? 🏡",
            "user_context": {},
            "intent": "CHITCHAT",
            "intent_related_response": "User is chitchatting",
            "results_to_save": [],
        }

        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type=HistoryType.onboarding
        )

        save_chat_history.assert_awaited_once_with(
            user_id="TestUser", messages=mock.ANY, history_type=HistoryType.onboarding
        )
        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 3
        assert saved_messages[0].text == "Hi!"
        assert saved_messages[1].text == "Hello!"
        assert (
            saved_messages[2].text == "Which province are you currently living in? 🏡"
        )


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA_TEXT", "Test Persona")
async def test_onboarding_first_question():
    """
    For the first interaction (no user input), we should get the first question
    and not perform intent detection or data extraction. The chat history should
    be initialized.
    """
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.delete_chat_history_for_user",
            new_callable=mock.AsyncMock,
        ) as delete_chat_history_for_user,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message"
        ) as handle_user_message,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
            return_value={
                "contextualized_question": "Which province are you currently living in? 🏡"
            },
        ) as _,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/onboarding",
            headers={"Authorization": "Token testtoken"},
            json={"user_id": "TestUser", "user_context": {}, "user_input": ""},
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Which province are you currently living in? 🏡",
            "user_context": {},
            "intent": "JOURNEY_RESPONSE",
            "intent_related_response": "",
            "results_to_save": [],
        }

        # Assert that history creation was attempted
        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type=HistoryType.onboarding
        )

        # Assert that the history deletion was then called
        delete_chat_history_for_user.assert_awaited_once_with(
            "TestUser", HistoryType.onboarding
        )

        # Assert that the new history is saved correctly
        save_chat_history.assert_awaited_once_with(
            user_id="TestUser", messages=mock.ANY, history_type=HistoryType.onboarding
        )
        saved_messages = save_chat_history.call_args.kwargs["messages"]
        # History should be initialized with the persona and the first question
        assert len(saved_messages) == 2
        assert saved_messages[0].is_from(role="system")
        assert saved_messages[0].text == "Test Persona"
        assert saved_messages[1].is_from(role="assistant")
        assert (
            saved_messages[1].text == "Which province are you currently living in? 🏡"
        )

        # These functions should not be called for the initial message
        handle_user_message.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_onboarding():
    """
    A standard user response should be processed, context updated, and
    the next question returned, with the interaction saved to history.
    """
    initial_history = [ChatMessage.from_assistant("Welcome!")]
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
            return_value=initial_history,
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message",
            return_value=("JOURNEY_RESPONSE", ""),
        ) as handle_user_message,
        mock.patch(
            "ai4gd_momconnect_haystack.api.process_onboarding_step",
            return_value=(
                {"area_type": "City"},
                {
                    "contextualized_question": "Which province are you currently living in? 🏡"
                },
            ),
        ) as mock_process_step,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/onboarding",
            headers={"Authorization": "Token testtoken"},
            json={"user_id": "TestUser", "user_context": {}, "user_input": "city"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Which province are you currently living in? 🏡",
            "user_context": {"area_type": "City"},
            "intent": "JOURNEY_RESPONSE",
            "intent_related_response": "",
            "results_to_save": ["area_type"],
        }

        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type=HistoryType.onboarding
        )

        save_chat_history.assert_awaited_once_with(
            user_id="TestUser", messages=mock.ANY, history_type=HistoryType.onboarding
        )
        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 3
        assert saved_messages[0].text == "Welcome!"
        assert saved_messages[1].text == "city"
        assert (
            saved_messages[2].text == "Which province are you currently living in? 🏡"
        )

        handle_user_message.assert_called_once()
        mock_process_step.assert_called_once_with(user_input="city", current_context={})


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_assessment_question",
    new_callable=mock.AsyncMock,
)
async def test_assessment_chitchat(
    save_assessment_question,
    validate_assessment_answer,
    get_assessment_question,
    handle_user_message,
):
    """
    Chitchat should give the user a specific chitchat message, and should not extract
    an answer, but should still ask the question again.
    """
    handle_user_message.return_value = "CHITCHAT", "User is chitchatting"
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
            "user_input": "Hello!",
            "flow_id": "dma-pre-assessment",
            "question_number": 1,
            "previous_question": "How are you feeling?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "next_question": 1,
        "intent": "CHITCHAT",
        "intent_related_response": "User is chitchatting",
        "processed_answer": None,
    }

    handle_user_message.assert_called_once_with("How are you feeling?", "Hello!")
    validate_assessment_answer.assert_not_called()
    get_assessment_question.assert_awaited_once()
    save_assessment_question.assert_awaited_once()


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
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "next_question": 1,
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
        "processed_answer": None,
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
    calculate_and_store_assessment_result,  # 2. Add it to the function signature
    get_assessment_question,
    handle_user_message,
):
    """
    On a valid user response, the endpoint should validate the answer, score it,
    save it, calculate the aggregate result, and then fetch and return the
    next question in the flow.
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
        question=None,
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
async def test_anc_survey():
    """
    Tests the ANC survey endpoint.
    It mocks the data extraction and question generation to ensure the API
    correctly processes the request and constructs the response.
    """
    initial_history = [
        ChatMessage.from_assistant(
            "Hi! Did you go for your clinic visit?",
            meta={"step_title": "clinic_visit_prompt"},
        )
    ]
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
            return_value=initial_history,
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message",
            return_value=("JOURNEY_RESPONSE", None),
        ) as _,
        mock.patch(
            "ai4gd_momconnect_haystack.api.extract_anc_data_from_response",
            return_value={"visit_status": "Yes, I went"},
        ) as _,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_anc_survey_question",
            return_value={
                "contextualized_question": "Great! Did you see a nurse or a doctor?",
                "is_final_step": False,
            },
        ) as _,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/survey",
            headers={"Authorization": "Token testtoken"},
            json={
                "user_id": "TestUser",
                "survey_id": "anc",
                "user_context": {},
                "user_input": "Yes I did",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Great! Did you see a nurse or a doctor?",
            "user_context": {"visit_status": "Yes, I went"},
            "survey_complete": False,
            "intent": "JOURNEY_RESPONSE",
            "intent_related_response": None,
            "results_to_save": ["visit_status"],
        }
        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type="anc"
        )
        assert save_chat_history.await_count == 2

        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 3
        assert saved_messages[0].text == "Hi! Did you go for your clinic visit?"
        assert saved_messages[1].text == "Yes, I went"
        assert saved_messages[2].text == "Great! Did you see a nurse or a doctor?"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA_TEXT", "Test Persona")
async def test_anc_survey_first_question():
    """
    For the first question, we shouldn't try to extract answers, and we shouldn't classify
    it as chitchat, even though it is blank because we don't have a user input
    """
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
            return_value=[],
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.delete_chat_history_for_user",
            new_callable=mock.AsyncMock,
        ) as delete_chat_history_for_user,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message"
        ) as handle_user_message,
        mock.patch(
            "ai4gd_momconnect_haystack.api.extract_anc_data_from_response"
        ) as extract_anc_data_from_response,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_anc_survey_question",
            return_value={
                "contextualized_question": "Hi! Did you go for your clinic visit?",
                "is_final_step": False,
            },
        ) as _,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/survey",
            headers={"Authorization": "Token testtoken"},
            json={
                "user_id": "TestUser",
                "survey_id": "anc",
                "user_context": {},
                "user_input": "",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Hi! Did you go for your clinic visit?",
            "user_context": {},
            "survey_complete": False,
            "intent": "JOURNEY_RESPONSE",
            "intent_related_response": "",
            "results_to_save": [],
        }

        # Assert that the history deletion was then called
        delete_chat_history_for_user.assert_awaited_once_with(
            "TestUser", HistoryType.anc
        )

        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type="anc"
        )
        assert save_chat_history.await_count == 1

        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 2
        assert saved_messages[0].is_from(role="system")
        assert saved_messages[0].text == "Test Persona"
        assert saved_messages[1].is_from(role="assistant")
        assert saved_messages[1].text == "Hi! Did you go for your clinic visit?"

        handle_user_message.assert_not_called()
        extract_anc_data_from_response.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_anc_survey_chitchat():
    """
    If the user sends chitchat, we should ask the same question again, and not
    try to extract an answer.
    """
    initial_history = [
        ChatMessage.from_assistant(
            "Hi! Did you go for your clinic visit?",
            meta={"step_title": "clinic_visit_prompt"},
        )
    ]
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_or_create_chat_history",
            new_callable=mock.AsyncMock,
            return_value=initial_history,
        ) as get_or_create_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message",
            return_value=("CHITCHAT", "User is chitchatting"),
        ) as _,
        mock.patch(
            "ai4gd_momconnect_haystack.api.extract_anc_data_from_response"
        ) as extract_anc_data_from_response,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_anc_survey_question",
            return_value={
                "contextualized_question": "Hi! Did you go for your clinic visit?",
                "is_final_step": False,
            },
        ) as _,
    ):
        client = TestClient(app)
        response = client.post(
            "/v1/survey",
            headers={"Authorization": "Token testtoken"},
            json={
                "user_id": "TestUser",
                "survey_id": "anc",
                "user_context": {},
                "user_input": "Hi!",
            },
        )

        assert response.status_code == 200
        assert response.json() == {
            "question": "Hi! Did you go for your clinic visit?",
            "user_context": {},
            "survey_complete": False,
            "intent": "CHITCHAT",
            "intent_related_response": "User is chitchatting",
            "results_to_save": [],
        }

        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type="anc"
        )
        assert save_chat_history.await_count == 1

        # The survey logic in api.py doesn't add the chitchat response to the
        # history, only the re-asked question.
        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 2
        assert saved_messages[0].text == "Hi! Did you go for your clinic visit?"
        assert saved_messages[1].text == "Hi! Did you go for your clinic visit?"

        extract_anc_data_from_response.assert_not_called()


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
