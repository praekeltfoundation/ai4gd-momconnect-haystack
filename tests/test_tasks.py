from unittest import mock
from ai4gd_momconnect_haystack.tasks import (
    get_assessment_question,
    validate_assessment_answer,
    get_next_onboarding_question,
    extract_onboarding_data_from_response,
)


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_assessment_question(pipelines_mock):
    pipelines_mock.run_assessment_contextualization_pipeline.return_value = (
        "mock_question"
    )
    result = get_assessment_question(
        flow_id="dma-assessment",
        current_assessment_step=0,
        user_context={},
    )
    assert result == {
        "contextualized_question": "mock_question",
        "current_question_number": 1,
    }


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_last_assessment_question(pipelines_mock):
    pipelines_mock.run_assessment_contextualization_pipeline.return_value = (
        "mock_question"
    )
    result = get_assessment_question(
        flow_id="dma-assessment",
        current_assessment_step=4,
        user_context={},
    )
    assert result == {
        "contextualized_question": "mock_question",
        "current_question_number": 5,
    }
    result = get_assessment_question(
        flow_id="dma-assessment",
        current_assessment_step=5,
        user_context={},
    )
    assert result == {}


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_assessment_question_invalid_flow(pipelines_mock):
    """
    Tests that get_assessment_question returns an empty dict for an invalid flow_id.
    """
    result = get_assessment_question(
        flow_id="non-existent-flow",
        current_assessment_step=0,
        user_context={},
    )
    assert result == {}
    pipelines_mock.run_assessment_contextualization_pipeline.assert_not_called()


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_validate_assessment_answer_success(pipelines_mock):
    """
    Tests successful validation of a user's response.
    """
    mock_processed_response = {"answer": "A", "is_valid": True}
    pipelines_mock.run_assessment_response_validator_pipeline.return_value = (
        mock_processed_response
    )

    result = validate_assessment_answer(
        user_response="This is my answer.", current_question_number=3
    )

    assert result == {
        "processed_user_response": mock_processed_response,
        "current_assessment_step": 3,
    }


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_validate_assessment_answer_failure(pipelines_mock):
    """
    Tests handling of an invalid user response where the pipeline returns nothing.
    """
    pipelines_mock.run_assessment_response_validator_pipeline.return_value = None

    result = validate_assessment_answer(
        user_response="I don't know.", current_question_number=3
    )

    # Check that the response is None and the step is decremented to repeat the question
    assert result == {
        "processed_user_response": None,
        "current_assessment_step": 2,
    }


@mock.patch("ai4gd_momconnect_haystack.tasks.doc_store")
@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_next_onboarding_question_no_more_questions(pipelines_mock, doc_store_mock):
    """
    Tests the scenario where all onboarding questions have been answered.
    """
    # Mock the function to return an empty list of remaining questions
    doc_store_mock.get_remaining_onboarding_questions.return_value = []

    result = get_next_onboarding_question(user_context={}, chat_history=[])
    assert result is None


@mock.patch("ai4gd_momconnect_haystack.tasks.doc_store")
@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_extract_onboarding_data_from_response_updates_context(
    pipelines_mock, doc_store_mock
):
    """
    Tests that the user_context is correctly updated with data
    extracted from the user's response.
    """
    # The initial state of the user context
    user_context = {"other": {}}

    # The data that the mocked pipeline will "extract"
    mock_extracted_data = {
        "province": "Gauteng",
        "education_level": "More than high school",
        "some_other_info": "extra detail",
    }

    pipelines_mock.run_onboarding_data_extraction_pipeline.return_value = (
        mock_extracted_data
    )

    # The user context that we expect after the function runs
    expected_context = {
        "province": "Gauteng",
        "education_level": "More than high school",
        "other": {"some_other_info": "extra detail"},
    }

    result_context = extract_onboarding_data_from_response(
        user_response="some response", user_context=user_context, chat_history=[]
    )

    assert result_context == expected_context
