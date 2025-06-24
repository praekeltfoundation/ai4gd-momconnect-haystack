import os
import pytest
from unittest import mock

from fastapi.testclient import TestClient
from haystack.dataclasses import ChatMessage
from sentry_sdk import get_client as get_sentry_client

from ai4gd_momconnect_haystack.api import app, setup_sentry
from ai4gd_momconnect_haystack.utilities import HistoryType


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
            return_value="Which province are you currently living in? 🏡",
        ) as _,
        mock.patch(
            "ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response",
            return_value={},
        ) as extract_onboarding_data_from_response,
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

        extract_onboarding_data_from_response.assert_not_called()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA", "Test Persona")
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
            "ai4gd_momconnect_haystack.api.save_chat_history",
            new_callable=mock.AsyncMock,
        ) as save_chat_history,
        mock.patch(
            "ai4gd_momconnect_haystack.api.handle_user_message"
        ) as handle_user_message,
        mock.patch(
            "ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response"
        ) as extract_onboarding_data_from_response,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
            return_value="Which province are you currently living in? 🏡",
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
        }

        # Assert that history creation was attempted
        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type=HistoryType.onboarding
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
        extract_onboarding_data_from_response.assert_not_called()


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
            "ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response",
            return_value={"area_type": "City"},
        ) as extract_onboarding_data_from_response,
        mock.patch(
            "ai4gd_momconnect_haystack.api.get_next_onboarding_question",
            return_value="Which province are you currently living in? 🏡",
        ) as get_next_onboarding_question,
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
        extract_onboarding_data_from_response.assert_called_once()
        get_next_onboarding_question.assert_called_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_pre_assessment_question",
    new_callable=mock.AsyncMock,
)
async def test_assessment_chitchat(
    save_pre_assessment_question,
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
    }

    handle_user_message.assert_called_once_with("How are you feeling?", "Hello!")
    validate_assessment_answer.assert_not_called()
    get_assessment_question.assert_awaited_once()
    save_pre_assessment_question.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch(
    "ai4gd_momconnect_haystack.api.get_assessment_question", new_callable=mock.AsyncMock
)
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
@mock.patch(
    "ai4gd_momconnect_haystack.api.save_pre_assessment_question",
    new_callable=mock.AsyncMock,
)
async def test_assessment_initial_message(
    save_pre_assessment_question,
    validate_assessment_answer,
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
    }

    handle_user_message.assert_not_called()
    validate_assessment_answer.assert_not_called()
    get_assessment_question.assert_awaited_once()
    save_pre_assessment_question.assert_awaited_once()


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
async def test_anc_survey():
    """
    Tests the ANC survey endpoint.
    It mocks the data extraction and question generation to ensure the API
    correctly processes the request and constructs the response.
    """
    initial_history = [
        ChatMessage.from_assistant("Hi! Did you go for your clinic visit?")
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
        }
        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type="anc"
        )
        assert save_chat_history.await_count == 1

        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 3
        assert saved_messages[0].text == "Hi! Did you go for your clinic visit?"
        assert saved_messages[1].text == "Yes I did"
        assert saved_messages[2].text == "Great! Did you see a nurse or a doctor?"


@pytest.mark.asyncio
@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.SERVICE_PERSONA", "Test Persona")
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
        }

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
        ChatMessage.from_assistant("Hi! Did you go for your clinic visit?")
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
        }

        get_or_create_chat_history.assert_awaited_once_with(
            user_id="TestUser", history_type="anc"
        )
        assert save_chat_history.await_count == 1

        # The survey logic in api.py doesn't add the chitchat response to the
        # history, only the re-asked question.
        saved_messages = save_chat_history.call_args.kwargs["messages"]
        assert len(saved_messages) == 3
        assert saved_messages[0].text == "Hi! Did you go for your clinic visit?"
        assert saved_messages[1].text == "Hi!"
        assert saved_messages[2].text == "Hi! Did you go for your clinic visit?"

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
