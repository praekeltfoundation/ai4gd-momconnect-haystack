from typing import Any
from unittest import mock

import pytest

# Import the Pydantic models and the functions to be tested
from ai4gd_momconnect_haystack.models import AssessmentRun, Question, Turn
from ai4gd_momconnect_haystack.tasks import (
    _calculate_assessment_score_range,
    _score_single_turn,
    extract_onboarding_data_from_response,
    get_assessment_question,
    get_next_onboarding_question,
    score_assessment_from_simulation,
    validate_assessment_answer,
)

# --- Test Data Fixtures ---


@pytest.fixture
def raw_assessment_questions() -> list[dict[str, Any]]:
    """Provides a sample list of raw question dictionaries."""
    return [
        {
            "question_number": 1,
            "question_name": "confident_in_making_health_decisions",
            "content": "Can you make decisions?",
            "valid_responses": {"No": 0, "A little": 1, "Yes": 2},
        },
        {
            "question_number": 2,
            "question_name": "confident_in_talking_to_health_worker",
            "content": "Can you talk to a worker?",
            "valid_responses": {"No": 0, "A little": 1, "Yes": 2},
        },
        {
            "question_number": 3,
            "question_name": "no_valid_responses",
            "content": "This question has no score options.",
            "valid_responses": {},  # Empty dict is valid, just not scorable
        },
    ]


@pytest.fixture
def validated_assessment_questions(raw_assessment_questions) -> list[Question]:
    """Provides a list of validated Pydantic Question models."""
    return [Question.model_validate(q) for q in raw_assessment_questions]


@pytest.fixture
def raw_simulation_output() -> list[dict[str, Any]]:
    """Provides a sample raw simulation output."""
    return [
        {
            "scenario_id": "dma_test_run",
            "flow_type": "dma-assessment",
            "turns": [
                # Valid turn
                {"question_number": 1, "llm_extracted_user_response": "Yes"},
                # Answer not in options
                {"question_number": 2, "llm_extracted_user_response": "Maybe"},
                # Question number doesn't exist in master file
                {"question_number": 99, "llm_extracted_user_response": "Yes"},
            ],
        }
    ]


@pytest.fixture
def validated_simulation_output(raw_simulation_output) -> list[AssessmentRun]:
    """Provides a list of validated Pydantic AssessmentRun models."""
    return [AssessmentRun.model_validate(run) for run in raw_simulation_output]


# --- Tests for the scoring logic ---


def test_calculate_assessment_score_range(validated_assessment_questions):
    """Tests the calculation of min and max possible scores."""
    min_score, max_score = _calculate_assessment_score_range(
        validated_assessment_questions
    )
    # Question 1: min 0, max 2
    # Question 2: min 0, max 2
    # Question 3: min 0, max 0
    # Total min: 0, Total max: 4
    assert min_score == 0
    assert max_score == 4


@pytest.mark.parametrize(
    "turn_data, expected_score, expected_error_substring",
    [
        ({"question_number": 1, "llm_extracted_user_response": "Yes"}, 2, None),
        (
            {"question_number": 2, "llm_extracted_user_response": "Maybe"},
            0,  # Score defaults to 0
            "not a valid, scorable option",
        ),
        (
            {"question_number": 99, "llm_extracted_user_response": "Yes"},
            0,  # Score defaults to 0
            "Question number 99 not found",  # UPDATED: More specific error check
        ),
    ],
    ids=[
        "success",
        "answer_not_in_options",
        "question_not_found",
    ],
)
def test_score_single_turn(
    turn_data, expected_score, expected_error_substring, validated_assessment_questions
):
    """Tests scoring a single turn. The input 'turn' is a validated Pydantic model."""
    turn_model = Turn.model_validate(turn_data)
    question_lookup = {q.question_number: q for q in validated_assessment_questions}

    # The function now returns a new dictionary instead of modifying in-place
    result = _score_single_turn(turn_model, question_lookup)

    assert result["score"] == expected_score
    if expected_error_substring:
        assert expected_error_substring in result["score_error"]
    else:
        # UPDATED: Safer check for missing key or a value of None.
        assert result.get("score_error") is None


def test_score_assessment_from_simulation_end_to_end(
    validated_simulation_output, validated_assessment_questions
):
    """Tests the full scoring process with validated Pydantic models."""
    result = score_assessment_from_simulation(
        simulation_output=validated_simulation_output,
        assessment_id="dma-assessment",
        assessment_questions=validated_assessment_questions,
    )

    assert result is not None
    assert result["overall_score"] == 2
    assert result["assessment_min_score"] == 0
    assert result["assessment_max_score"] == 4
    assert result["score_percentage"] == 50.0

    # Check that the final output has a clean 'turns' list
    results_list = result["turns"]
    assert len(results_list) == 3
    assert results_list[0]["score"] == 2
    assert results_list[1]["score"] == 0  # Score defaults to 0 for invalid answers
    assert "not a valid, scorable option" in results_list[1]["score_error"]


def test_score_assessment_from_simulation_no_valid_run(validated_assessment_questions):
    """Tests that the function returns None if the assessment_id is not found."""
    simulation_output_missing_flow = [
        AssessmentRun.model_validate(
            {"scenario_id": "s1", "flow_type": "onboarding", "turns": []}
        )
    ]

    result = score_assessment_from_simulation(
        simulation_output=simulation_output_missing_flow,
        assessment_id="dma-assessment",
        assessment_questions=validated_assessment_questions,
    )
    assert result is None


# --- Tests for the question fetching logic ---


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
        user_response="This is my answer.",
        current_question_number=3,
        current_flow_id="dma-assessment",
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
        user_response="I don't know.",
        current_question_number=3,
        current_flow_id="dma-assessment",
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
    user_context = {}

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
