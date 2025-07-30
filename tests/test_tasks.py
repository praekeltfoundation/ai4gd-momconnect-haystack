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
    ResponseScore,
    Turn,
)
from ai4gd_momconnect_haystack.tasks import (
    extract_onboarding_data_from_response,
    get_assessment_question,
    get_next_onboarding_question,
    handle_intro_response,
    handle_conversational_repair,
    extract_anc_data_from_response,
    handle_summary_confirmation_step,
    format_user_data_summary_for_whatsapp,
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


@pytest.fixture
def sample_user_context():
    """Provides a sample user_context after data collection is complete."""
    return {
        "province": "Gauteng",
        "area_type": "City",
        "relationship_status": "Single",
    }


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


@pytest.mark.parametrize(
    "mock_intent, mock_validation, expected_action",
    [
        ("JOURNEY_RESPONSE", "Yes", "PROCEED"),
        ("JOURNEY_RESPONSE", "No", "ABORT"),
        ("JOURNEY_RESPONSE", None, "REPROMPT"),  # Ambiguous response
        ("ASKING_TO_STOP_MESSAGES", None, "ABORT"),
        ("QUESTION_ABOUT_STUDY", None, "REPROMPT_WITH_ANSWER"),
        ("CHITCHAT", None, "ABORT"),
    ],
    ids=[
        "Consents Yes",
        "Consents No",
        "Ambiguous Consent",
        "Asks to Stop",
        "Asks a Question",
        "Chitchat",
    ],
)
def test_handle_intro_response(mock_intent, mock_validation, expected_action):
    """
    Tests that the handle_intro_response task returns the correct action
    based on the classified intent and validated response.
    """
    with (
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.handle_user_message",
            return_value=(mock_intent, "mock response"),
        ) as mock_handle_msg,
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.pipelines.run_clinic_visit_data_extraction_pipeline",
            return_value=mock_validation,
        ) as mock_run_pipeline,
    ):
        result = handle_intro_response(user_input="test input", flow_id="onboarding")

        assert result["action"] == expected_action
        mock_handle_msg.assert_called_once()

        # The validation pipeline should only be called if the intent is JOURNEY_RESPONSE
        if mock_intent == "JOURNEY_RESPONSE":
            mock_run_pipeline.assert_called_once()
        else:
            mock_run_pipeline.assert_not_called()


@pytest.mark.parametrize(
    "pipeline_return, expected_substring",
    [
        ("Rephrased question from LLM", "Rephrased question from LLM"),
        (None, "Sorry, I didn't understand"),  # Test the fallback
    ],
    ids=["pipeline_success", "pipeline_failure_fallback"],
)
def test_handle_conversational_repair(pipeline_return, expected_substring):
    """
    Tests the handle_conversational_repair function for both successful
    pipeline execution and fallback behavior.
    """
    mock_question = AssessmentQuestion(
        question_number=1,
        content="Original Question?",
        valid_responses_and_scores=[ResponseScore(response="Yes", score=1)],
    )

    with (
        mock.patch(
            "ai4gd_momconnect_haystack.tasks.run_rephrase_question_pipeline",
            return_value=pipeline_return,
        ) as mock_run_pipeline,
        mock.patch.dict(
            "ai4gd_momconnect_haystack.tasks.assessment_flow_map",
            {"test-flow": [mock_question]},
        ),
    ):
        result = handle_conversational_repair(
            flow_id="test-flow",
            question_identifier=1,
            previous_question="Original Question?",
            invalid_input="bad answer",
        )

        assert expected_substring in result
        mock_run_pipeline.assert_called_once_with(
            previous_question="Original Question?",
            invalid_input="bad answer",
            valid_responses=["Yes"],
        )


@mock.patch.dict(
    "ai4gd_momconnect_haystack.tasks.ANC_SURVEY_MAP",
    {"start": mock.Mock(content="Q1", valid_responses=["Yes, I went"])},
)
@mock.patch(
    "ai4gd_momconnect_haystack.tasks.pipelines.run_survey_data_extraction_pipeline"
)
def test_extract_anc_data_high_confidence(mock_run_pipeline):
    """Tests that a high-confidence match correctly updates the context."""
    mock_run_pipeline.return_value = {
        "validated_response": "Yes, I went",
        "match_type": "exact",
        "confidence": "high",
    }

    context, action_dict = extract_anc_data_from_response(
        user_response="yebo", user_context={}, step_title="start"
    )

    assert context["start"] == "Yes, I went"
    assert action_dict is None


@mock.patch.dict(
    "ai4gd_momconnect_haystack.tasks.ANC_SURVEY_MAP",
    {"start": mock.Mock(content="Q1", valid_responses=["I'm going soon"])},
)
@mock.patch(
    "ai4gd_momconnect_haystack.tasks.pipelines.run_survey_data_extraction_pipeline"
)
def test_extract_anc_data_low_confidence_triggers_clarification(mock_run_pipeline):
    """Tests that a low-confidence match returns an action_dict to trigger the clarification loop."""
    mock_run_pipeline.return_value = {
        "validated_response": "I'm going soon",
        "match_type": "inferred",
        "confidence": "low",
    }

    context, action_dict = extract_anc_data_from_response(
        user_response="not yet", user_context={}, step_title="start"
    )

    assert "start" not in context
    assert action_dict is not None
    assert action_dict["status"] == "needs_confirmation"
    assert action_dict["potential_answer"] == "I'm going soon"


@mock.patch.dict(
    "ai4gd_momconnect_haystack.tasks.ANC_SURVEY_MAP",
    {
        "Q_challenges": mock.Mock(
            content="Q_challenges",
            valid_responses=["Transport 🚌", "Something else 😞"],
        )
    },
)
@mock.patch(
    "ai4gd_momconnect_haystack.tasks.pipelines.run_survey_data_extraction_pipeline"
)
def test_extract_anc_data_no_match_handles_other(mock_run_pipeline):
    """Tests that a no_match response correctly populates the 'other' fields in the context."""
    mock_run_pipeline.return_value = {
        "validated_response": "I was too sick",
        "match_type": "no_match",
        "confidence": "high",
    }

    context, action_dict = extract_anc_data_from_response(
        user_response="I was too sick", user_context={}, step_title="Q_challenges"
    )

    assert context["Q_challenges"] == "Something else 😞"
    assert context["Q_challenges_other_text"] == "I was too sick"
    assert action_dict is None


def test_format_user_data_summary_for_whatsapp(sample_user_context):
    """
    Tests the WhatsApp summary formatting function for correctness.
    """
    # Scenario 1: Full context
    summary = format_user_data_summary_for_whatsapp(sample_user_context)
    assert "Here's the information I have for you:" in summary
    assert "_*Province*: Gauteng_" in summary
    assert "_*Area Type*: City_" in summary
    assert "_*Relationship Status*: Single_" in summary
    assert "Is this all correct?" in summary

    # Scenario 2: Context with a skipped value should not show the skipped field
    context_with_skip = {"province": "Limpopo", "area_type": "Skipped - System"}
    summary_with_skip = format_user_data_summary_for_whatsapp(context_with_skip)
    assert "_*Province*: Limpopo_" in summary_with_skip
    assert "Area Type" not in summary_with_skip


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines.run_data_update_pipeline")
def test_handle_summary_confirmation_step_with_update(
    mock_run_pipeline, sample_user_context
):
    """
    Tests the summary handler when the user provides an update.
    """
    # Mock the pipeline to return an update
    mock_run_pipeline.return_value = {"province": "KwaZulu-Natal"}

    user_input = "my province is KZN"
    context_copy = sample_user_context.copy()
    result = handle_summary_confirmation_step(user_input, context_copy)

    mock_run_pipeline.assert_called_once_with(user_input, context_copy)
    assert result["intent"] == "ONBOARDING_UPDATE_COMPLETE"
    assert "Thank you! I've updated your information." in result["question"]
    assert result["results_to_save"] == ["province"]

    updated_context = result["user_context"]
    assert updated_context["province"] == "KwaZulu-Natal"
    assert "flow_state" not in updated_context


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines.run_data_update_pipeline")
def test_handle_summary_confirmation_step_with_confirmation(
    mock_run_pipeline, sample_user_context
):
    """
    Tests the summary handler when the user confirms their data (no updates extracted).
    """
    # Mock the pipeline to return no updates
    mock_run_pipeline.return_value = {}

    user_input = "yes, that is correct"
    context_copy = sample_user_context.copy()
    result = handle_summary_confirmation_step(user_input, context_copy)

    mock_run_pipeline.assert_called_once_with(user_input, context_copy)
    assert result["intent"] == "ONBOARDING_COMPLETE"
    assert "Perfect, thank you! Your onboarding is complete." in result["question"]
    assert "flow_state" not in result["user_context"]


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines.run_data_update_pipeline")
def test_handle_summary_confirmation_step_with_denial(
    mock_run_pipeline, sample_user_context
):
    """
    Tests the summary handler when the user denies the summary and must be
    re-prompted for clarification.
    """
    # Mock the pipeline to return no updates, simulating a simple "no"
    mock_run_pipeline.return_value = {}

    user_input = "no that is wrong"
    # The context must still have the flow_state for this test
    context_with_state = sample_user_context.copy()
    context_with_state["flow_state"] = "confirming_summary"

    result = handle_summary_confirmation_step(user_input, context_with_state)

    # Assert the pipeline was called
    mock_run_pipeline.assert_called_once_with(user_input, context_with_state)

    # Assert the result is a repair/re-prompt
    assert result["intent"] == "REPAIR"
    assert "Please tell me what you would like to change" in result["question"]

    # Assert that the flow_state was NOT removed, to allow the conversation to continue
    updated_context = result["user_context"]
    assert updated_context.get("flow_state") == "confirming_summary"
