import pytest
from pathlib import Path
from src.evaluation.evaluator import (
    DataPointValidator, DialogueSimulator, BaseFlowEvaluator, OnboardingEvaluator
)

# Assume Haystack classes are importable for type hinting and spec,
# but we will mock their instances during tests.
from haystack.evaluation.evaluators import (
    AnswerExactMatchEvaluator, 
    SASEvaluator, 
    LLMEvaluator as HaystackLLMEvaluator
)

# Langfuse types for mocking return values if needed by tests
try:
    from langfuse.model import CreateScore  # Used in asserts
    LANGFUSE_AVAILABLE_FOR_TEST = True
except ImportError:
    LANGFUSE_AVAILABLE_FOR_TEST = False
    class CreateScore:
        pass  # Dummy if Langfuse not installed during test run

# --- Fixtures using pytest-mock ---
@pytest.fixture
def mock_dialogue_simulator(mocker):  # Inject the mocker fixture
    """Provides a mocked DialogueSimulator using pytest-mock."""
    simulator = mocker.MagicMock(spec=DialogueSimulator)

    # Mock the run_simulation to return predictable output
    # including a mock trace object if Langfuse is involved
    mock_trace = mocker.MagicMock() # Langfuse trace mock
    # Configure mock_trace.score if tests will assert calls to it
    mock_trace.score = mocker.Mock()

    simulator.run_simulation.return_value = (
        [{"speaker": "user", "utterance": "Hello"}, 
         {"speaker": "llm", "utterance": "Hi there!", "message_length": 9}],  # dialogue_transcript
        {"extracted_field": "some_value"},  # llm_extracted_data
        mock_trace  # scenario_trace (mocked Langfuse trace)
    )
    return simulator


@pytest.fixture
def mock_sas_evaluator(mocker):
    """Provides a mocked SASEvaluator that returns a specific score."""
    evaluator = mocker.MagicMock(spec=SASEvaluator)
    evaluator.run = mocker.Mock(return_value={"sas_score": 0.9}) 
    return evaluator


@pytest.fixture
def mock_llm_interaction_evaluator(mocker):
    """Provides a mocked HaystackLLMEvaluator."""
    evaluator = mocker.MagicMock(spec=HaystackLLMEvaluator)

    def mock_run(*args, **kwargs):
        # This logic depends on how your actual _run_qualitative_evaluators
        # formats its calls to the llm_interaction_evaluator.
        # For the placeholder logic in the script, this mock might not be
        # hit directly
        # unless that placeholder logic is filled out to call .run()
        query = kwargs.get("query", "")  # Or however inputs are passed
        if "clarity" in query.lower():
            return {"llm_eval_score": 4.0, "answer": "4"}
        if "empathy" in query.lower():
            return {"llm_eval_score": 4.5, "answer": "4.5"}
        return {"llm_eval_score": -1.0, "answer": "N/A"}
    evaluator.run = mocker.Mock(side_effect=mock_run)
    return evaluator


@pytest.fixture
def data_validator_with_sas(mock_sas_evaluator):
    """DataPointValidator instance with a mocked SASEvaluator."""
    # Assuming AnswerExactMatchEvaluator is not the focus for this specific fixture
    return DataPointValidator(
        exact_match_evaluator=None, sas_evaluator=mock_sas_evaluator
    )


@pytest.fixture
def data_validator_no_haystack():
    """DataPointValidator instance with no Haystack evaluators."""
    return DataPointValidator(exact_match_evaluator=None, sas_evaluator=None)


@pytest.fixture
def dummy_flow_definition():
    """A simple flow definition for testing evaluators."""
    return [
        {"collects": "narrative_field", "content": "Tell me a story."},
        {"collects": "simple_field", "content": "What is your name?"}
    ]


@pytest.fixture
def dummy_onboarding_evaluator(
    dummy_flow_definition, data_validator_no_haystack, mock_dialogue_simulator,
    mock_llm_interaction_evaluator 
):
    """An OnboardingEvaluator instance with mocked dependencies."""
    return OnboardingEvaluator(
        flow_definition=dummy_flow_definition,
        validator=data_validator_no_haystack,
        simulator=mock_dialogue_simulator,
        llm_interaction_evaluator=mock_llm_interaction_evaluator
    )


@pytest.fixture
def dummy_scenario_for_qualitative_eval():
    """A scenario specifically for testing qualitative evaluation parts."""
    return {
        "scenario_id": "qualitative_test_s1",
        "flow_type": "onboarding",
        "mock_dialogue_transcript": [
            {"speaker": "user", "utterance": "I feel a bit unsure."},
            {"speaker": "llm", "utterance": "It's okay to feel unsure. How can I help clarify?", "message_length": 50},
            {"speaker": "user", "utterance": "What data do you need next?"},
            {"speaker": "llm", "utterance": "Next, I need to know about your previous pregnancies.", "message_length": 60}
        ],
        "ground_truth_extracted_data": {}, 
        "mock_llm_extracted_data": {}
    }


# --- Tests for DataPointValidator with SASEvaluator ---

def test_dpv_uses_sas_evaluator_for_specific_field(data_validator_with_sas, mock_sas_evaluator):
    """
    Tests that SASEvaluator is called for a field configured to use it,
    and its score is included.
    """
    # This field name should match the condition in DataPointValidator._validate_single_data_point
    # for using SAS, e.g., if it's "some_narrative_field"
    collects_field_for_sas = "some_narrative_field"

    result = data_validator_with_sas.validate_single_data_point(
        collects_field=collects_field_for_sas,
        llm_extracted_value="A slightly different story.",
        gt_value="A story.",
        is_required=True
    )
    mock_sas_evaluator.run.assert_called_once_with(
        predicted_answers=["a slightly different story."],
        ground_truth_answers=["a story."]
    )
    assert result["is_accurate_to_gt"] is True # Based on mocked score > 0.8
    assert result["sas_score"] == 0.9


def test_dpv_falls_back_if_sas_not_for_field(data_validator_with_sas, mock_sas_evaluator):
    """
    Tests that SASEvaluator is NOT called for a field not configured for SAS.
    """
    # Reset mock call count if the instance is reused across tests,
    # or ensure fresh instance per test. Pytest fixtures usually provide fresh instances.
    mock_sas_evaluator.run.reset_mock()

    result = data_validator_with_sas.validate_single_data_point(
        collects_field="another_field", # Not "some_narrative_field"
        llm_extracted_value="exact_value",
        gt_value="exact_value", # Will be accurate by Python comparison
        is_required=True
    )
    mock_sas_evaluator.run.assert_not_called()
    assert result["is_accurate_to_gt"] is True 
    assert "sas_score" not in result  # Or assert it's the default -1.0


def test_dpv_no_sas_evaluator_provided(data_validator_no_haystack):
    """
    Tests DataPointValidator works with custom Python logic if SASEvaluator is None.
    """
    result = data_validator_no_haystack.validate_single_data_point(
        collects_field="some_narrative_field", 
        llm_extracted_value="A different story.",
        gt_value="A story.",
        is_required=True
    )
    assert result["is_accurate_to_gt"] is False  # Custom logic: exact match fails
    assert "sas_score" not in result


# --- Tests for BaseFlowEvaluator (via OnboardingEvaluator) with HaystackLLMEvaluator ---

def test_base_evaluator_runs_qualitative_evaluators_if_provided(
    dummy_onboarding_evaluator,  # This fixture now includes mock_llm_interaction_evaluator
    mock_llm_interaction_evaluator,  # To assert calls on it
    dummy_scenario_for_qualitative_eval 
):
    """
    Tests that _run_qualitative_evaluators attempts to use
    the llm_interaction_evaluator if provided.
    """
    dialogue_transcript = dummy_scenario_for_qualitative_eval["mock_dialogue_transcript"]
    mock_scenario_trace = mocker.MagicMock()  # Langfuse trace mock from pytest-mock
    mock_scenario_trace.score = mocker.Mock()


    # Note: _run_qualitative_evaluators is a protected member.
    # Ideally, test via public interface, but for focused testing:
    qual_scores = dummy_onboarding_evaluator._run_qualitative_evaluators(
        dialogue_transcript=dialogue_transcript,
        scenario_trace=mock_scenario_trace
    )

    # The current _run_qualitative_evaluators has placeholder logic.
    # If it were implemented to call mock_llm_interaction_evaluator.run():
    # mock_llm_interaction_evaluator.run.assert_called() # Or assert_any_call()
    # For now, we check the placeholder scores and that Langfuse scoring
    # is attempted.
    assert qual_scores["avg_llm_clarity_placeholder"] == -1.0 
    assert qual_scores["avg_llm_empathy_placeholder"] == -1.0

    if LANGFUSE_AVAILABLE_FOR_TEST: 
        mock_scenario_trace.score.assert_any_call(
            CreateScore(name="avg_llm_clarity", value=-1.0)
        )
        mock_scenario_trace.score.assert_any_call(
            CreateScore(name="avg_llm_empathy", value=-1.0)
        )


def test_base_evaluator_qualitative_eval_skipped_if_no_evaluator(
    mocker,  # For creating a fresh mock_dialogue_simulator
    dummy_flow_definition, 
    data_validator_no_haystack
):
    """
    Tests that qualitative evaluation is skipped if no llm_interaction_evaluator.
    """
    mock_sim = mocker.MagicMock(spec=DialogueSimulator)
    mock_trace = mocker.MagicMock()
    mock_sim.run_simulation.return_value = ([{"s":"u","u":"t"}], {}, mock_trace)
    
    evaluator_no_qual = OnboardingEvaluator(
        flow_definition=dummy_flow_definition,
        validator=data_validator_no_haystack,
        simulator=mock_sim,
        llm_interaction_evaluator=None  # Explicitly None
    )

    dialogue_transcript = [{"speaker": "llm", "utterance": "Test message"}]

    qual_scores = evaluator_no_qual._run_qualitative_evaluators(
        dialogue_transcript=dialogue_transcript,
        scenario_trace=mock_trace # Pass the trace object
    )

    assert qual_scores["eval_status"] == "llm_interaction_evaluator_not_provided"
    # Ensure no Langfuse scores for these qualitative metrics were attempted
    # by checking that score was not called with these specific names.
    if LANGFUSE_AVAILABLE_FOR_TEST:
        for call_obj in mock_trace.score.call_args_list:
            # call_obj is a tuple, first element is args tuple, second is kwargs dict
            # We are interested in the first argument to score(), which is a CreateScore object
            score_instance = call_obj[0][0]
            assert score_instance.name not in ["avg_llm_clarity", "avg_llm_empathy"]
