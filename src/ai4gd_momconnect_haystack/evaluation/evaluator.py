"""
A script to evaluate LLM performance based on three criteria for each turn:
1.  Question Consistency: Is the LLM's question similar to the ground truth question?
2.  User Response Appropriateness: Is the user's answer relevant to the question?
3.  Extraction Accuracy: Does the extracted data exactly match the ground truth?
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric
from deepeval.test_case import LLMTestCase

# ==============================================================================
# UPDATED SECTION 1: Import the main simulation function
# ==============================================================================
# This allows the evaluator to call the main simulation script directly.
from ai4gd_momconnect_haystack.main import async_main as run_main_simulation

# Configure logging to show only important messages
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class Colors:
    """Simple ANSI color codes for terminal output."""

    OK = "\033[92m"  # GREEN
    WARNING = "\033[93m"  # YELLOW
    FAIL = "\033[91m"  # RED
    BOLD = "\033[1m"  # BOLD
    ENDC = "\033[0m"  # RESET


def load_json_file(file_path: Path) -> list[dict] | dict | None:
    """Loads a JSON file, returning its content or None on error."""
    if not file_path.exists():
        logging.error(f"File not found at '{file_path}'.")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from '{file_path}'.")
        return None


def _get_metric_details(result: Any) -> tuple[float, str | None]:
    """
    Safely extracts the score and reason from a deepeval result,
    providing default values if they don't exist.
    """
    score = 0.0
    reason = None
    if (
        result.test_results
        and result.test_results[0].metrics_data
        and result.test_results[0].metrics_data[0]
    ):
        metric_data = result.test_results[0].metrics_data[0]
        if metric_data.score is not None:
            score = metric_data.score
        if metric_data.reason is not None:
            reason = metric_data.reason
    return score, reason


def _preprocess_text(text: Any) -> str:
    """
    Applies post-processing to a string to normalize it for evaluation.
    - Converts to lowercase.
    - Removes emojis.
    - Strips leading/trailing whitespace.
    """
    if not isinstance(text, str):
        text = str(text)  # Ensure it's a string

    # Convert to lowercase
    text = text.lower()

    # Remove emojis using a regex
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    # Strip leading/trailing whitespace and remove extra spaces
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def run_evaluation_suite(
    gt_path: Path,
    results_path: Path,
    eval_model: str,
    threshold: float,
    report_path: Path | None = None,
):
    """
    Loads data, runs a three-part evaluation suite, and prints reports.
    """
    gt_data = load_json_file(gt_path)
    llm_results_data = load_json_file(results_path)

    gt_scenarios = []
    if isinstance(gt_data, dict):
        gt_scenarios = gt_data.get("scenarios", [])
    elif isinstance(gt_data, list):
        gt_scenarios = gt_data

    llm_results = llm_results_data if isinstance(llm_results_data, list) else []

    if not gt_scenarios or not llm_results:
        exit(
            "Evaluation aborted: could not load or parse ground truth or results data."
        )

    gt_lookup: dict[tuple, dict] = {}
    for s in gt_scenarios:
        for t in s.get("turns", []):
            identifier_key = (
                "question_name"
                if "onboarding" in s["flow_type"] or "anc-survey" in s["flow_type"]
                else "question_number"
            )
            identifier = t.get(identifier_key)
            if identifier:
                gt_lookup[(s["scenario_id"], s["flow_type"], identifier)] = t

    llm_results_lookup: dict[tuple, dict] = {}
    for s in llm_results:
        for t in s.get("turns", []):
            identifier_key = (
                "question_name"
                if "onboarding" in s["flow_type"] or "anc-survey" in s["flow_type"]
                else "question_number"
            )
            identifier = t.get(identifier_key)
            if identifier:
                llm_results_lookup[(s["scenario_id"], s["flow_type"], identifier)] = t

    collated_results: dict[tuple, dict] = {}
    relevancy_metric = AnswerRelevancyMetric(
        threshold=threshold, model=eval_model, include_reason=True
    )
    metrics_list: list[BaseMetric] = [relevancy_metric]

    print("\n--- Running Per-Turn Evaluations (this may take a moment) ---")

    for key, gt_turn in gt_lookup.items():
        llm_result_turn = llm_results_lookup.get(key)
        if not llm_result_turn:
            logging.warning(f"No LLM result found for key: {key}")
            continue

        try:
            gt_llm_utterance = _preprocess_text(gt_turn.get("llm_utterance", ""))
            llm_utterance = _preprocess_text(llm_result_turn.get("llm_utterance", ""))
            gt_user_utterance = _preprocess_text(gt_turn.get("user_utterance", ""))

            q_consistency_result = evaluate(
                test_cases=[
                    LLMTestCase(input=gt_llm_utterance, actual_output=llm_utterance)
                ],
                metrics=metrics_list,
            )
            r_appropriateness_result = evaluate(
                test_cases=[
                    LLMTestCase(input=llm_utterance, actual_output=gt_user_utterance)
                ],
                metrics=metrics_list,
            )

            expected_extraction_str = _preprocess_text(gt_turn.get("user_response", ""))
            actual_extraction_str = _preprocess_text(
                llm_result_turn.get("user_response", "")
            )
            exact_match_passed = actual_extraction_str == expected_extraction_str

            q_score, q_reason = _get_metric_details(q_consistency_result)
            r_score, r_reason = _get_metric_details(r_appropriateness_result)

            collated_results[key] = {
                "exact_match_passed": exact_match_passed,
                "extraction_details": f"Expected: '{expected_extraction_str}', Got: '{actual_extraction_str}'",
                "question_consistency_score": q_score,
                "question_consistency_reason": q_reason,
                "response_appropriateness_score": r_score,
                "response_appropriateness_reason": r_reason,
            }
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during evaluation for turn {key}: {e}"
            )
    present_results(collated_results, output_file=report_path)


def present_results(all_results: dict[tuple, dict], output_file: Path | None = None):
    """Formats and prints reports and optionally saves them to a file."""

    report_lines: list[str] = []

    # --- Build Detailed Report ---
    def add_line(console_line: str, file_line: str | None = None):
        """Helper to print to console and add a clean version to the report list."""
        print(console_line)
        report_lines.append(file_line if file_line is not None else console_line)

    header = "=" * 50
    title = "              DETAILED TURN-BY-TURN REPORT"
    add_line("\n\n" + header, "\n" + header)
    add_line(title)
    add_line(header)

    for (scenario_id, flow_type, turn_identifier), res in all_results.items():
        turn_header = (
            f"--- Turn: {scenario_id} ({flow_type}) | ID: {turn_identifier} ---"
        )
        add_line(f"\n{Colors.BOLD}{turn_header}{Colors.ENDC}", f"\n{turn_header}")

        extraction_details = res.get("extraction_details", "No details available.")
        if res.get("exact_match_passed"):
            add_line(
                f"  {Colors.OK}[✅] Extraction Accuracy: PASSED{Colors.ENDC}",
                "  [✅] Extraction Accuracy: PASSED",
            )
        else:
            add_line(
                f"  {Colors.FAIL}[❌] Extraction Accuracy: FAILED{Colors.ENDC}",
                "  [❌] Extraction Accuracy: FAILED",
            )
            add_line(
                f"       {Colors.FAIL}Details: {extraction_details}{Colors.ENDC}",
                f"       Details: {extraction_details}",
            )

        qc_score = res.get("question_consistency_score", 0.0)
        qc_reason = res.get("question_consistency_reason")
        qc_color = Colors.OK if qc_score >= 0.7 else Colors.FAIL
        add_line(
            f"  {qc_color}[ score: {qc_score:.2f} ] Question Consistency{Colors.ENDC}",
            f"  [ score: {qc_score:.2f} ] Question Consistency",
        )
        if qc_reason and qc_score < 0.7:
            add_line(
                f"       {Colors.WARNING}Reason: {qc_reason}{Colors.ENDC}",
                f"       Reason: {qc_reason}",
            )

        ra_score = res.get("response_appropriateness_score", 0.0)
        ra_reason = res.get("response_appropriateness_reason")
        ra_color = Colors.OK if ra_score >= 0.7 else Colors.FAIL
        add_line(
            f"  {ra_color}[ score: {ra_score:.2f} ] Response Appropriateness{Colors.ENDC}",
            f"  [ score: {ra_score:.2f} ] Response Appropriateness",
        )
        if ra_reason and ra_score < 0.7:
            add_line(
                f"       {Colors.WARNING}Reason: {ra_reason}{Colors.ENDC}",
                f"       Reason: {ra_reason}",
            )

    # --- Build Summary Report ---
    summary_data: dict[str, list] = {}
    for (scenario_id, flow_type, question_name), res in all_results.items():
        if res:
            summary_data.setdefault(flow_type, []).append(res)

    summary_header = "=" * 50
    summary_title = "              PERFORMANCE SUMMARY REPORT"
    add_line("\n\n" + summary_header, "\n\n" + summary_header)
    add_line(summary_title)
    add_line(summary_header)

    for flow_type, results_list in summary_data.items():
        if not results_list:
            continue
        total = len(results_list)
        ea_rate = sum(1 for r in results_list if r["exact_match_passed"]) / total
        qc_avg = sum(r["question_consistency_score"] for r in results_list) / total
        ra_avg = sum(r["response_appropriateness_score"] for r in results_list) / total

        flow_header = (
            f"📊 {flow_type.capitalize()} Performance ({total} turns evaluated):"
        )
        add_line(f"\n{Colors.BOLD}{flow_header}{Colors.ENDC}", f"\n{flow_header}")
        add_line(f"  - Extraction Accuracy: {ea_rate:.2%}")
        add_line(f"  - Question Consistency Score (Avg): {qc_avg:.2f}")
        add_line(f"  - Response Appropriateness Score (Avg): {ra_avg:.2f}")

    add_line("\n" + summary_header)

    # --- Save Report to File ---
    if output_file:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            add_line(
                f"\n{Colors.OK}Evaluation report successfully saved to: {output_file}{Colors.ENDC}",
                f"\nEvaluation report successfully saved to: {output_file}",
            )
        except IOError as e:
            logging.error(f"Could not write report to file '{output_file}': {e}")


def main():
    """Parses arguments and orchestrates the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run a multi-faceted evaluation on LLM results."
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="First run the automated simulation to generate a fresh results file.",
    )
    # ==============================================================================
    # UPDATED SECTION: Improved help text for clarity on file naming.
    # ==============================================================================
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save the final evaluation report to a text file. The report will be named after the results file "
        "(e.g., 'results.json' -> 'results.report.txt').",
    )
    parser.add_argument(
        "--gt-file", type=Path, default=script_dir / "data/ground_truth.json"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Path to a specific LLM results file to evaluate. If not provided, --run-simulation must be used.",
    )
    parser.add_argument(
        "--openai-key", type=str, default=os.environ.get("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--deepeval-key", type=str, default=os.environ.get("DEEPEVAL_API_KEY")
    )
    parser.add_argument("--eval-model", type=str, default="gpt-4o")
    parser.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()

    if not args.openai_key:
        logging.error(
            "OpenAI API key not found. Provide it via --openai-key or set OPENAI_API_KEY."
        )
        return
    os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.deepeval_key:
        os.environ["DEEPEVAL_API_KEY"] = args.deepeval_key

    results_path_to_evaluate = args.results_file

    if args.run_simulation:
        if args.results_file:
            logging.warning(
                "Both --run-simulation and --results-file were provided. "
                "Ignoring --results-file and generating a new simulation run."
            )

        print("--- Running Automated Simulation ---")
        output_file_path = asyncio.run(
            run_main_simulation(
                GT_FILE_PATH=args.gt_file,
                RESULT_PATH=script_dir / "data",
                is_automated=True,
                save_simulation=True,
            )
        )

        if output_file_path and output_file_path.exists():
            print(
                f"--- Simulation complete. Using generated file: {output_file_path} ---"
            )
            results_path_to_evaluate = output_file_path
        else:
            logging.error(
                "Simulation run failed or did not produce an output file. Aborting evaluation."
            )
            return

    if not results_path_to_evaluate:
        logging.error(
            "No results file specified. Use --results-file <path> or add --run-simulation to generate one."
        )
        return

    report_path = None
    if args.save_report:
        report_path = results_path_to_evaluate.with_suffix(".report.txt")
        # ==============================================================================
        # NEW SECTION: Add a print statement for user confirmation.
        # ==============================================================================
        print(f"--- Evaluation report will be saved to: {report_path} ---")

    run_evaluation_suite(
        gt_path=args.gt_file,
        results_path=results_path_to_evaluate,
        eval_model=args.eval_model,
        threshold=args.threshold,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
