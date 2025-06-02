# tests/evaluation_suite/conftest.py
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock # Can still use MagicMock directly or via mocker

# Assuming your classes are importable
from src.evaluation.core_utils import CoreUtils, OutputConfiguration
from src.evaluation.evaluator import (
    DialogueSimulator, DataPointValidator, OnboardingEvaluator, AssessmentEvaluator
)


# Haystack and Langfuse mocks if needed by fixtures directly
try:
    from langfuse import Langfuse
except ImportError:
    class Langfuse: pass

try:
    from haystack.components.evaluators  import (
        AnswerExactMatchEvaluator, 
        SASEvaluator, 
        LLMEvaluator as HaystackLLMEvaluator
    )
except ImportError:
    class AnswerExactMatchEvaluator: pass
    class SASEvaluator: pass
    class HaystackLLMEvaluator: pass


@pytest.fixture
def mock_output_configuration(tmp_path: Path) -> OutputConfiguration:
    """Provides an OutputConfiguration instance using a temporary path."""
    return OutputConfiguration(base_path=tmp_path / "eval_outputs_test")


@pytest.fixture
def mock_dialogue_simulator(mocker) -> DialogueSimulator:
    """Provides a generic mocked DialogueSimulator."""
    simulator = mocker.MagicMock(spec=DialogueSimulator)
    mock_trace = mocker.MagicMock() # Mock for Langfuse trace
    mock_trace.score = mocker.Mock() # Ensure score method on trace is mockable
    simulator.run_simulation.return_value = (
        [{"speaker": "user", "utterance": "Test input"}], # mock dialogue_transcript
        {"extracted_key": "mock_value"}, # mock llm_extracted_data
        mock_trace # mock scenario_trace
    )
    return simulator


@pytest.fixture
def mock_data_validator(mocker) -> DataPointValidator:
    """Provides a generic mocked DataPointValidator."""
    validator = mocker.MagicMock(spec=DataPointValidator)
    validator.validate_single_data_point.return_value = {
        "collects_field": "test_field", "llm_value_extracted": "val",
        "gt_value": "val", "is_collected": True, "is_accurate_to_gt": True,
        "is_valid_option": True, "is_required_and_missing": False
    }
    validator.analyze_dialogue_metrics.return_value = {
        "total_user_inputs": 1, "total_llm_responses": 1,
        "user_hesitancy": {"skips": 0, "why_asks": 0},
        "llm_msg_size": {"avg_len": 10, "max_len": 10, "min_len": 10, 
                         "over_thresh": 0, "thresh": 400}
    }
    # Mock _normalize_value if it's called directly by evaluator logic
    validator._normalize_value = mocker.Mock(
        side_effect=lambda val,
        field=None: str(val).lower().strip()
        if val is not None else None
    )
    return validator


@pytest.fixture
def mock_llm_interaction_evaluator(mocker) -> HaystackLLMEvaluator | None:
    """Provides a mocked HaystackLLMEvaluator for qualitative checks."""
    # Return None if Haystack is not intended to be a hard dependency for all
    # tests or if you want to test pathways where it's not provided.
    # For testing its integration, return a mock:
    evaluator = mocker.MagicMock(spec=HaystackLLMEvaluator)
    evaluator.run.return_value = {"mock_qual_score": 4.5} # Example output
    return evaluator
    # return None # To test scenarios where it's not passed


@pytest.fixture
def dummy_onboarding_flow_def() -> list[dict[str, Any]]:
    return [
        {"question_number": 1, "content": "Province?", "collects": "province", "valid_responses": ["Gauteng", "KZN", "Skip"]},
        {"question_number": 2, "content": "Area type?", "collects": "area_type", "valid_responses": ["City", "Rural"]},
    ]


@pytest.fixture
def dummy_assessment_flow_def() -> list[dict[str, Any]]:
    return [
        {"question_number": 1, "content": "Confidence health decisions?", "collects": "conf_health", "valid_responses": ["Confident", "Not confident"]},
        {"question_number": 2, "content": "Confidence discuss medical?", "collects": "conf_discuss", "valid_responses": ["Confident", "Not confident"]},
    ]


@pytest.fixture
def happy_onboarding_scenario() -> dict[str, Any]:
    return {
        "scenario_id": "onboarding_happy_s1",
        "flow_type": "onboarding",
        "mock_dialogue_transcript": [
            {"speaker": "user", "utterance": "I live in Gauteng"},
            {"speaker": "llm", "utterance": "Okay, Gauteng. And the area type?", "message_length": 40}
        ],
        "ground_truth_extracted_data": {"province": "Gauteng", "area_type": "City"},
        "mock_llm_extracted_data": {"province": "Gauteng", "area_type": "City"}
    }


@pytest.fixture
def happy_assessment_scenario() -> dict[str, Any]:
    return {
        "scenario_id": "assessment_happy_s1",
        "flow_type": "dma-assessment",
        "mock_dialogue_transcript": [
            {"speaker": "llm", "utterance": "Confidence in health decisions?", "message_length": 30},
            {"speaker": "user", "utterance": "Confident"},
            {"speaker": "llm", "utterance": "Confidence in discussing medical problems?", "message_length": 40},
            {"speaker": "user", "utterance": "Not confident"},
        ],
        "ground_truth_extracted_data": {"conf_health": "Confident", "conf_discuss": "Not confident"},
        "mock_llm_extracted_data": {"conf_health": "Confident", "conf_discuss": "Not confident"},
        "mock_llm_collection_sequence": ["conf_health", "conf_discuss"] # Crucial for order check
    }