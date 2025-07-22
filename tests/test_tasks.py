from typing import Any
from unittest import mock

import pytest

# Import the Pydantic models and the functions to be tested
from ai4gd_momconnect_haystack.assessment_logic import (
    _calculate_assessment_score_range,
    _score_single_turn,
    score_assessment_from_simulation,
    validate_assessment_answer,
)
from ai4gd_momconnect_haystack.enums import AssessmentType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentQuestion,
    AssessmentRun,
    Turn,
)
from ai4gd_momconnect_haystack.tasks import (
    extract_onboarding_data_from_response,
    get_assessment_question,
    get_next_onboarding_question,
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
            "valid_responses_and_scores": [
                {"response": "No", "score": 0},
                {"response": "A little", "score": 1},
                {"response": "Yes", "score": 2},
            ],
        },
        {
            "question_number": 2,
            "question_name": "confident_in_talking_to_health_worker",
            "content": "Can you talk to a worker?",
            "valid_responses_and_scores": [
                {"response": "No", "score": 0},
                {"response": "A little", "score": 1},
                {"response": "Yes", "score": 2},
            ],
        },
        {
            "question_number": 3,
            "question_name": "no_valid_responses",
            "content": "This question has no score options.",
            "valid_responses_and_scores": [],
        },
    ]


@pytest.fixture
def validated_assessment_questions(
    raw_assessment_questions,
) -> list[AssessmentQuestion]:
    """Provides a list of validated Pydantic Question models."""
    return [AssessmentQuestion.model_validate(q) for q in raw_assessment_questions]


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
            0,
            "not a valid, scorable option",
        ),
        (
            {"question_number": 99, "llm_extracted_user_response": "Yes"},
            0,
            "Question number 99 not found",
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
        assessment_id="dma-pre-assessment",
        assessment_questions=validated_assessment_questions,
    )
    assert result is None


# --- Tests for the question fetching logic ---


@pytest.mark.asyncio
async def test_get_assessment_question():
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.pipelines.run_assessment_contextualization_pipeline",
            return_value="mock_question",
        ) as mock_run_pipeline,
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.get_assessment_history",
            new_callable=mock.AsyncMock,
            return_value=[],
        ) as mock_get_history,
    ):
        result = await get_assessment_question(
            user_id="TestUser",
            flow_id=AssessmentType.dma_pre_assessment,
            question_number=1,
            user_context={},
        )
        assert result == {
            "contextualized_question": "mock_question\n\na. I strongly disagree\nb. I disagree\nc. I'm not sure\nd. I agree\ne. I strongly agree",
        }

        mock_run_pipeline.assert_called_once()
        mock_get_history.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_last_assessment_question():
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.pipelines.run_assessment_contextualization_pipeline",
            return_value="mock_question",
        ) as mock_run_pipeline,
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.get_assessment_history",
            new_callable=mock.AsyncMock,
            return_value=[],
        ) as mock_get_history,
    ):
        # This call should succeed and use the mocks.
        result = await get_assessment_question(
            user_id="TestUser",
            flow_id=AssessmentType.dma_pre_assessment,
            question_number=5,
            user_context={},
        )
        assert result == {
            "contextualized_question": "mock_question\n\na. I strongly disagree\nb. I disagree\nc. I'm not sure\nd. I agree\ne. I strongly agree",
        }

        # This call should fail before the pipeline is even created or run,
        # because the question number is out of bounds.
        result = await get_assessment_question(
            user_id="TestUser",
            flow_id=AssessmentType.dma_pre_assessment,
            question_number=6,
            user_context={},
        )
        assert result == {}

        mock_run_pipeline.assert_called_once()
        mock_get_history.assert_awaited()


@mock.patch(
    "ai4gd_momconnect_haystack.tasks.pipelines.run_assessment_response_validator_pipeline"
)
def test_validate_assessment_answer_success(mock_run_pipeline):
    """
    Tests successful validation of a user's response.
    """
    mock_processed_response = "A"
    mock_run_pipeline.return_value = mock_processed_response

    with mock.patch(
        "ai4gd_momconnect_haystack.tasks.pipelines.create_assessment_response_validator_pipeline"
    ):
        result = validate_assessment_answer(
            user_response="This is my answer.",
            question_number=3,
            current_flow_id="dma-pre-assessment",
        )

    assert result == {
        "processed_user_response": mock_processed_response,
        "next_question_number": 4,
    }

    mock_run_pipeline.assert_called_once()


@mock.patch(
    "ai4gd_momconnect_haystack.tasks.pipelines.run_assessment_response_validator_pipeline"
)
def test_validate_assessment_answer_failure(mock_run_pipeline):
    """
    Tests handling of an invalid user response where the pipeline returns nothing.
    """
    mock_run_pipeline.return_value = None

    with mock.patch(
        "ai4gd_momconnect_haystack.tasks.pipelines.create_assessment_response_validator_pipeline"
    ):
        result = validate_assessment_answer(
            user_response="I don't know.",
            question_number=3,
            current_flow_id="dma-pre-assessment",
        )

    assert result == {
        "processed_user_response": None,
        "next_question_number": 3,
    }

    mock_run_pipeline.assert_called_once()


@mock.patch("ai4gd_momconnect_haystack.tasks.doc_store")
@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_next_onboarding_question_no_more_questions(pipelines_mock, doc_store_mock):
    """
    Tests the scenario where all onboarding questions have been answered.
    """
    # Mock the function to return an empty list of remaining questions
    doc_store_mock.get_remaining_onboarding_questions.return_value = []

    result = get_next_onboarding_question(user_context={})
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
        "some_other_info": "extra detail",
    }

    result_context = extract_onboarding_data_from_response(
        user_response="some response",
        user_context=user_context,
        current_question="Some question text",
    )

    assert result_context == expected_context
