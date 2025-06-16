import os
from unittest import mock

from fastapi.testclient import TestClient
from sentry_sdk import get_client as get_sentry_client

from ai4gd_momconnect_haystack.api import app, setup_sentry


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


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response")
def test_onboarding_chitchat(
    extract_onboarding_data_from_response,
    get_next_onboarding_question,
    handle_user_message,
):
    """
    If the user's response gets classified as chitchat, then we should not try
    to extract the onboarding data. Also, the response to the chitchat should be
    captured in the message history
    """
    extract_onboarding_data_from_response.return_value = {"area_type": "City"}
    get_next_onboarding_question.return_value = (
        "Which province are you currently living in? 🏡"
    )
    handle_user_message.return_value = "CHITCHAT", "User is chitchatting"
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_context": {}, "user_input": "Hello!"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "Which province are you currently living in? 🏡",
        "user_context": {},
        "chat_history": [
            "User to System: Hello!",
            "System to User: User is chitchatting",
            "System to User: Which province are you currently living in? 🏡",
        ],
        "intent": "CHITCHAT",
        "intent_related_response": "User is chitchatting",
    }
    extract_onboarding_data_from_response.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response")
def test_onboarding_first_question(
    extract_onboarding_data_from_response,
    get_next_onboarding_question,
    handle_user_message,
):
    """
    For the first interaction that the user has with the service, we don't have a user
    input. That should be handled correctly and not seen as chitchat
    """
    extract_onboarding_data_from_response.return_value = {"area_type": "City"}
    get_next_onboarding_question.return_value = (
        "Which province are you currently living in? 🏡"
    )
    handle_user_message.return_value = "CHITCHAT", "User is chitchatting"
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_context": {}, "user_input": ""},
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "Which province are you currently living in? 🏡",
        "user_context": {},
        "chat_history": [
            "System to User: Which province are you currently living in? 🏡",
        ],
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    }
    handle_user_message.assert_not_called()
    extract_onboarding_data_from_response.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response")
def test_onboarding(
    extract_onboarding_data_from_response,
    get_next_onboarding_question,
    handle_user_message,
):
    """
    For the first interaction that the user has with the service, we don't have a user
    input. That should be handled correctly and not seen as chitchat
    """
    extract_onboarding_data_from_response.return_value = {"area_type": "City"}
    get_next_onboarding_question.return_value = (
        "Which province are you currently living in? 🏡"
    )
    handle_user_message.return_value = "JOURNEY_RESPONSE", ""
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_context": {}, "user_input": "city"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "Which province are you currently living in? 🏡",
        "user_context": {"area_type": "City"},
        "chat_history": [
            "User to System: city",
            "System to User: Which province are you currently living in? 🏡",
        ],
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    }


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_question")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
def test_assessment_chitchat(
    validate_assessment_answer, get_assessment_question, handle_user_message
):
    """
    Chitchat should give the user a specific chitchat message, and should not extract
    an answer, but should still ask the question again
    """
    get_assessment_question.return_value = {
        "contextualized_question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "current_question_number": 1,
    }
    handle_user_message.return_value = "CHITCHAT", "User is chitchatting"
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_context": {},
            "user_input": "Hello!",
            "flow_id": "dma-assessment",
            "question_number": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "next_question": 1,
        "chat_history": [
            "User to System: Hello!",
            "System to User: User is chitchatting",
            "System to User: How confident are you in discussing your maternal health concerns with your healthcare provider?",
        ],
        "intent": "CHITCHAT",
        "intent_related_response": "User is chitchatting",
    }
    validate_assessment_answer.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_question")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
def test_assessment_initial_message(
    validate_assessment_answer, get_assessment_question, handle_user_message
):
    """
    On the first interaction with the user, we don't have a user input. This should be
    treated as a Journey Response and we should not check the intent or extract an answer
    """
    get_assessment_question.return_value = {
        "contextualized_question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "current_question_number": 2,
    }
    client = TestClient(app)
    response = client.post(
        "/v1/assessment",
        headers={"Authorization": "Token testtoken"},
        json={
            "user_context": {},
            "user_input": "",
            "flow_id": "dma-assessment",
            "question_number": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "next_question": 2,
        "chat_history": [
            "System to User: How confident are you in discussing your maternal health concerns with your healthcare provider?"
        ],
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    }
    handle_user_message.assert_not_called()
    validate_assessment_answer.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_anc_survey_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_anc_data_from_response")
def test_anc_survey(
    extract_anc_data_from_response, get_anc_survey_question, handle_user_message
):
    """
    Tests the ANC survey endpoint.
    It mocks the data extraction and question generation to ensure the API
    correctly processes the request and constructs the response.
    """
    extract_anc_data_from_response.return_value = {"visit_status": "Yes, I went"}
    get_anc_survey_question.return_value = {
        "contextualized_question": "Great! Did you see a nurse or a doctor?",
        "is_final_step": False,
    }
    handle_user_message.return_value = "JOURNEY_RESPONSE", None
    client = TestClient(app)
    initial_chat_history = ["System to User: Hi! Did you go for your clinic visit?"]

    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "survey_id": "anc",
            "user_context": {},
            "user_input": "Yes I did",
            "chat_history": initial_chat_history,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Great! Did you see a nurse or a doctor?",
        "user_context": {"visit_status": "Yes, I went"},
        "chat_history": [
            "System to User: Hi! Did you go for your clinic visit?",
            "User to System: Yes I did",
            "System to User: Great! Did you see a nurse or a doctor?",
        ],
        "survey_complete": False,
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": None,
    }


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_anc_survey_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_anc_data_from_response")
def test_anc_survey_first_question(
    extract_anc_data_from_response, get_anc_survey_question, handle_user_message
):
    """
    For the first question, we shouldn't try to extract answers, and we shouldn't classify
    it as chitchat, even though it is blank because we don't have a user input
    """
    get_anc_survey_question.return_value = {
        "contextualized_question": "Hi! Did you go for your clinic visit?",
        "is_final_step": False,
    }
    client = TestClient(app)

    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "survey_id": "anc",
            "user_context": {},
            "user_input": "",
            "chat_history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Hi! Did you go for your clinic visit?",
        "user_context": {},
        "chat_history": [
            "System to User: Hi! Did you go for your clinic visit?",
        ],
        "survey_complete": False,
        "intent": "JOURNEY_RESPONSE",
        "intent_related_response": "",
    }
    handle_user_message.assert_not_called()
    extract_anc_data_from_response.assert_not_called()


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.handle_user_message")
@mock.patch("ai4gd_momconnect_haystack.api.get_anc_survey_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_anc_data_from_response")
def test_anc_survey_chitchat(
    extract_anc_data_from_response, get_anc_survey_question, handle_user_message
):
    """
    If the user sends chitchat, we should ask the same question again, and not
    try to extract an answer.
    """
    handle_user_message.return_value = "CHITCHAT", "User is chitchatting"
    get_anc_survey_question.return_value = {
        "contextualized_question": "Hi! Did you go for your clinic visit?",
        "is_final_step": False,
    }
    client = TestClient(app)

    response = client.post(
        "/v1/survey",
        headers={"Authorization": "Token testtoken"},
        json={
            "survey_id": "anc",
            "user_context": {},
            "user_input": "Hi!",
            "chat_history": [
                "System to User: Hi! Did you go for your clinic visit?",
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Hi! Did you go for your clinic visit?",
        "user_context": {},
        "chat_history": [
            "System to User: Hi! Did you go for your clinic visit?",
            "User to System: Hi!",
            "System to User: User is chitchatting",
            "System to User: Hi! Did you go for your clinic visit?",
        ],
        "survey_complete": False,
        "intent": "CHITCHAT",
        "intent_related_response": "User is chitchatting",
    }
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
            "survey_id": "invalid",
            "user_context": {},
            "user_input": "Hi",
            "chat_history": [],
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "ctx": {"expected": "'anc'"},
                "input": "invalid",
                "loc": ["body", "survey_id"],
                "msg": "Input should be 'anc'",
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
