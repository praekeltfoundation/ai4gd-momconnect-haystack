import json
import datetime
from typing import Any
from pathlib import Path

# For production system, will replacing print statements
# for errors/warnings with a more structured logging framework


class CoreUtils:
    """Utility functions for core operations like file I/O."""

    @staticmethod
    def load_json_file(
        filepath: Path, description: str = "file"
    ) -> Any | None:
        """
        Loads data from a JSON file using pathlib.Path.

        Args:
            filepath: Path to the JSON file.
            description: Description of the file for error messages.

        Returns:
            Parsed JSON data or None if an error occurs.
        """
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        try:
            with filepath.open('r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {description.capitalize()} file not found "
                  f"at {filepath.resolve()}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {filepath.resolve()} "
                  f"({description}): {e}")
            return None
        except Exception as e:  # pylint: disable=broad-except
            print(f"An unexpected error occurred while loading "
                  f"{filepath.resolve()}: {e}")
            return None

    @staticmethod
    def load_golden_dataset(filepath: Path) -> list[dict[str, Any]]:
        """
        Loads the golden dataset (list of scenarios) from a JSON file.
        Returns an empty list if loading fails or data is not a list.
        """
        data = CoreUtils.load_json_file(filepath, "golden dataset")
        return data if isinstance(data, list) else []

    @staticmethod
    def load_flow_definitions(
        onboarding_flows_path: Path, assessment_flows_path: Path
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """
        Loads the ONBOARDING_FLOWS and ASSESSMENT_FLOWS from JSON files.
        Returns a tuple of (onboarding_flows, assessment_flows), where
        either can be None if loading failed or structure is invalid.
        """
        onboarding_flows = CoreUtils.load_json_file(
            onboarding_flows_path, "onboarding flows"
        )
        assessment_flows = CoreUtils.load_json_file(
            assessment_flows_path, "assessment flows"
        )
        
        # Validate basic expected structure
        if not (
            onboarding_flows and
            isinstance(onboarding_flows.get("onboarding"), list)
        ):
            print(f"Warning: 'onboarding' key is missing or its value "
                  f"is not a list in {onboarding_flows_path.resolve()}. "
                  f"Onboarding flows may not be usable.")
            onboarding_flows = None
        if not (
            assessment_flows and
            isinstance(assessment_flows.get("dma-assessment"), list)
        ):
            print(f"Warning: 'dma-assessment' key is missing or its value "
                  f"is not a list in {assessment_flows_path.resolve()}. "
                  f"Assessment flows may not be usable.")
            assessment_flows = None
        return onboarding_flows, assessment_flows

    @staticmethod
    def save_report(report_data: dict[str, Any], filepath: Path):
        """Saves a report (dictionary) to a JSON file."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open('w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4)
            print(f"Report saved to {filepath.resolve()}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error saving report to {filepath.resolve()}: {e}")


class OutputConfiguration:
    """Manages output paths for evaluation reports for a single run."""
    def __init__(self, base_path: str | Path = "evaluation_outputs"):
        """
        Initializes output paths, creating a timestamped run directory.

        Args:
            base_path: The root directory for all evaluation outputs.
        """
        self.base_path: Path = Path(base_path).resolve()
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_path: Path = self.base_path / timestamp

        self.onboarding_summary_path: Path = \
            self.run_path / "onboarding_summary.json"
        self.assessment_summary_path: Path = \
            self.run_path / "assessment_summary.json"
        self.detailed_reports_path: Path = \
            self.run_path / "detailed_scenarios"

        # Create directories
        try:
            self.detailed_reports_path.mkdir(parents=True, exist_ok=True)
            print(f"Outputs for this run will be saved under {self.run_path}")
        except OSError as e:
            print(f"Error creating output directories at {self.run_path}: {e}")
            # Potentially raise the error or handle it if directory creation is critical