import json
import pytest
from pathlib import Path
from src.evaluation.core_utils import CoreUtils, OutputConfiguration


def test_load_json_file_success(tmp_path: Path):
    """Tests successful loading of a JSON file."""
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.json"
    test_data = {"key": "value", "number": 123}
    with p.open("w", encoding="utf-8") as f:
        json.dump(test_data, f)

    loaded_data = CoreUtils.load_json_file(p, "test json")
    assert loaded_data == test_data


def test_load_json_file_not_found(tmp_path: Path):
    """Tests behavior when JSON file is not found."""
    p = tmp_path / "non_existent.json"
    loaded_data = CoreUtils.load_json_file(p, "non-existent json")
    assert loaded_data is None


def test_load_json_file_invalid_json(tmp_path: Path):
    """Tests behavior with an invalid JSON file."""
    p = tmp_path / "invalid.json"
    with p.open("w", encoding="utf-8") as f:
        f.write("{'key': 'value',,}") # Invalid JSON

    loaded_data = CoreUtils.load_json_file(p, "invalid json")
    assert loaded_data is None


def test_output_configuration_paths(tmp_path: Path):
    """Tests if OutputConfiguration creates paths correctly."""
    base_output_path = tmp_path / "test_outputs"
    output_conf = OutputConfiguration(base_output_path)

    assert output_conf.base_path == base_output_path.resolve()
    assert output_conf.run_path.parent == base_output_path.resolve()
    assert output_conf.detailed_reports_path.exists()
    assert "detailed_scenarios" in str(output_conf.detailed_reports_path)
    assert "onboarding_summary.json" in str(output_conf.onboarding_summary_path)

# TODO:  add more tests for load_golden_dataset, load_flow_definitions,
# save_report ...
