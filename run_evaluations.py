# run_my_evaluations.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv # If you use dotenv for local runs outside Docker

# Optional: Add src to sys.path if it's not automatically found,
# though running from project root usually makes 'src' importable.
# SCRIPT_DIR = Path(__file__).resolve().parent
# SRC_DIR = SCRIPT_DIR / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# Import the main function from your evaluator script
# Python needs to find the 'src' directory. If running this script from project_root,
# 'src' should be discoverable.
from src.evaluation.evaluator import main_evaluation_runner, Langfuse, AnswerExactMatchEvaluator, SASEvaluator, HaystackLLMEvaluator


def main():
    print("Starting evaluations via top-level runner script...")
    load_dotenv()  # Load .env file if present (for local runs)

    # --- Define paths to your data files (relative to this script or absolute) ---
    # Assuming this script (run_my_evaluations.py) is at the project root.
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "src" / "evaluation" / "data"  # As per your dummy data example

    # Or, if your actual data is elsewhere, point to it:
    # data_dir = project_root / "evaluation_data_actual"

    dummy_golden_fp = data_dir / "golden_dataset.json"
    onboarding_flows_fp = data_dir / "onboarding_flow.json" # Should point to your actual file
    assessment_flows_fp = data_dir / "assessment_flow.json" # Should point to your actual file

    # Initialize Langfuse client (example, same as in your evaluator's __main__)
    langfuse_client = None
    try:
        # Ensure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST are set
        # (e.g., from .env loaded by load_dotenv() or set in environment)
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        if not langfuse_public_key: # Check if key is loaded
            print("Langfuse public key not found in environment. Langfuse disabled.")
        else:
            langfuse_client = Langfuse() # Will use env vars
            if langfuse_client.auth_check():
                print("Langfuse client initialized and authenticated successfully from top runner.")
            else:
                print("Langfuse authentication failed in top runner. Tracing will be disabled.")
                langfuse_client = None
    except Exception as e:
        print(f"Failed to initialize Langfuse in top runner: {e}. Tracing disabled.")

    # Initialize Haystack evaluators (example)
    haystack_exact_match_eval = None
    try:
        haystack_exact_match_eval = AnswerExactMatchEvaluator()
    except Exception:
        pass # Handle if not available or needed

    haystack_sas_eval = None # Initialize as needed
    haystack_llm_q_eval = None # Initialize as needed

    # Call the main function from your evaluator module
    main_evaluation_runner(
        golden_dataset_path=dummy_golden_fp,    # Or your actual golden dataset path
        onboarding_flows_json_path=onboarding_flows_fp,
        assessment_flows_json_path=assessment_flows_fp,
        langfuse_instance=langfuse_client,
        exact_match_evaluator_instance=haystack_exact_match_eval,
        sas_evaluator_instance=haystack_sas_eval,
        llm_interaction_eval_instance=haystack_llm_q_eval
    )


if __name__ == "__main__":
    main()