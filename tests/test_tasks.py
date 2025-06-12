import pytest
from unittest import mock
from ai4gd_momconnect_haystack.tasks import (
    get_assessment_question,
    _score_and_format_turn,
    _calculate_assessment_score_range,
    score_assessment_from_simulation
)

# --- Test Data Fixtures ---


@pytest.fixture
def mock_assessment_questions():
    """Provides a sample list of assessment questions for scoring tests."""
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
            "question_name": "has_invalid_score_value",
            "content": "This question has a bad score value.",
            "valid_responses": {"Yes": "two", "No": 0},  # Contains an invalid score
        },
        {
            "question_number": 4,
            "question_name": "no_valid_responses",
            "content": "This question has no score options.",
        },
    ]


@pytest.fixture
def mock_simulation_output():
    """Provides a sample simulation output for scoring tests."""
    return [
        {
            "flow_type": "dma_assessment",
            "turns": [
                # Valid turn
                {"question_name": 1, "llm_extracted_user_response": "Yes"},
                # Answer not in options
                {"question_name": 2, "llm_extracted_user_response": "Maybe"},
                # Question number doesn't exist in master file
                {"question_name": 99, "llm_extracted_user_response": "Yes"},
                # Malformed turn (missing identifier)
                {
                    "some_other_key": "value",
                    "llm_extracted_user_response": "Yes",
                },
            ],
        }
    ]


@pytest.fixture
def mock_dma_flow_for_get_question():
    """Provides a minimal DMA flow for get_assessment_question tests."""
    return [
        {"question_number": 1, "content": "Q1"},
        {"question_number": 2, "content": "Q2"},
        {"question_number": 3, "content": "Q3"},
        {"question_number": 4, "content": "Q4"},
        {"question_number": 5, "content": "Q5"},
    ]


# --- Tests for the scoring logic ---


def test_calculate_assessment_score_range(mock_assessment_questions):
    """Tests the calculation of min and max possible scores."""
    min_score, max_score = _calculate_assessment_score_range(mock_assessment_questions)
    # Question 1: min 0, max 2
    # Question 2: min 0, max 2
    # Question 3: min 0, max 0 (only "No" is a valid int)
    # Question 4: min 0, max 0 (no score options)
    # Total min: 0, Total max: 4
    assert min_score == 0
    assert max_score == 4


@pytest.mark.parametrize(
    "turn_input, expected_score, expected_error_substring",
    [
        ({"question_name": 1, "llm_extracted_user_response": "Yes"}, 2, None),
        (
            {"question_name": 2, "llm_extracted_user_response": "Maybe"},
            None,
            "not a valid, scorable option",
        ),
        (
            {"question_name": 99, "llm_extracted_user_response": "Yes"},
            None,
            "Question not found",
        ),
        (
            {"llm_extracted_user_response": "Yes"},
            None,
            "missing its question identifier",
        ),
        (
            {"question_name": 3, "llm_extracted_user_response": "Yes"},
            None,
            "Invalid score value",
        ),
    ],
    ids=[
        "success",
        "answer_not_in_options",
        "question_not_found",
        "missing_identifier",
        "invalid_score_value",
    ],
)
def test_score_and_format_turn(
    turn_input, expected_score, expected_error_substring, mock_assessment_questions
):
    """Tests various scenarios for scoring a single turn using parametrization."""
    question_lookup = {q["question_number"]: q for q in mock_assessment_questions}
    
    _score_and_format_turn(turn_input, question_lookup)

    assert turn_input["score"] == expected_score
    if expected_error_substring:
        assert expected_error_substring in turn_input["score_error"]
    else:
        assert turn_input["score_error"] is None


def test_score_assessment_from_simulation_end_to_end(
    mock_simulation_output, mock_assessment_questions
):
    """Tests the full scoring process and the new output format."""
    import copy
    simulation_output_copy = copy.deepcopy(mock_simulation_output)

    result = score_assessment_from_simulation(
        simulation_output=simulation_output_copy,
        assessment_id="dma_assessment",
        assessment_questions=mock_assessment_questions,
    )

    assert result is not None
    assert result["overall_score"] == 2
    assert result["assessment_min_score"] == 0
    assert result["assessment_max_score"] == 4
    assert result["score_percentage"] == 50.0

    # Check that original turns are present and augmented
    results_list = result["results"]
    assert len(results_list) == 4
    assert results_list[0]["score"] == 2
    assert results_list[1]["score"] is None


def test_score_assessment_from_simulation_no_valid_run(mock_assessment_questions):
    """Tests that the function returns None if the assessment_id is not found."""
    simulation_output_missing_flow = [{"flow_type": "onboarding", "turns": []}]

    result = score_assessment_from_simulation(
        simulation_output=simulation_output_missing_flow,
        assessment_id="dma_assessment",
        assessment_questions=mock_assessment_questions,
    )
    assert result is None


# --- Tests for the question fetching logic ---


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_assessment_question(pipelines_mock):
    pipelines_mock.run_assessment_contextualization_pipeline.return_value = (
        "mock_question"
    )
    result = get_assessment_question(
        flow_id="assessment_flow_id",
        question_number=0,
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
        flow_id="assessment_flow_id",
        question_number=4,
        current_assessment_step=4,
        user_context={},
    )
    assert result == {
        "contextualized_question": "mock_question",
        "current_question_number": 5,
    }
    result = get_assessment_question(
        flow_id="assessment_flow_id",
        question_number=5,
        current_assessment_step=5,
        user_context={},
    )
    assert result == {}
