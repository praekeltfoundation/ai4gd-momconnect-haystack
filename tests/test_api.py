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
@mock.patch("ai4gd_momconnect_haystack.api.get_next_onboarding_question")
@mock.patch("ai4gd_momconnect_haystack.api.extract_onboarding_data_from_response")
def test_onboarding(
    extract_onboarding_data_from_response, get_next_onboarding_question
):
    extract_onboarding_data_from_response.return_value = {"area_type": "City"}
    get_next_onboarding_question.return_value = (
        "Which province are you currently living in? 🏡"
    )
    client = TestClient(app)
    response = client.post(
        "/v1/onboarding",
        headers={"Authorization": "Token testtoken"},
        json={"user_context": {}, "user_input": "Hello!"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "user_context": {"area_type": "City"},
        "question": "Which province are you currently living in? 🏡",
        "chat_history": [
            "User to System: Hello!",
            "System to User: Which province are you currently living in? 🏡",
        ],
    }


@mock.patch.dict(os.environ, {"API_TOKEN": "testtoken"}, clear=True)
@mock.patch("ai4gd_momconnect_haystack.api.get_assessment_question")
@mock.patch("ai4gd_momconnect_haystack.api.validate_assessment_answer")
def test_assessment(validate_assessment_answer, get_assessment_question):
    get_assessment_question.return_value = {
        "contextualized_question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
        "current_question_number": 2,
    }
    validate_assessment_answer.return_value = {
        "processed_user_response": "testresponse",
        "current_assessment_step": 1,
    }
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
        "next_question": 2,
        "chat_history": [
            "User to System: Hello!",
            "System to User: How confident are you in discussing your maternal health concerns with your healthcare provider?",
        ],
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
