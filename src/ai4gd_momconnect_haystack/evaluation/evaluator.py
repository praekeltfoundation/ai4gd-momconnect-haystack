"""
A script to evaluate LLM performance based on three criteria for each turn:
1.  Question Consistency: Is the LLM's question similar to the ground truth question?
2.  User Response Appropriateness: Is the user's answer relevant to the question?
3.  Extraction Accuracy: Does the extracted data exactly match the ground truth?
"""

import os
import json
import logging
import argparse
from pathlib import Path

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Configure logging to show only important messages
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def load_json_file(file_path: Path) -> list[dict] | None:
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


def run_evaluation_suite(
    gt_path: Path, results_path: Path, eval_model: str, threshold: float
):
    """
    Loads data, runs a three-part evaluation suite, and prints reports.
    """
    gt_scenarios = load_json_file(gt_path)
    llm_results = load_json_file(results_path)

    if gt_scenarios is None or llm_results is None:
        exit("Evaluation aborted due to file loading errors.")

    # Create lookup dictionaries for efficient matching
    gt_lookup = {
        (s["scenario_id"], s["flow_type"], t["question_name"]): t
        for s in gt_scenarios
        for t in s["turns"]
    }
    llm_results_lookup = {
        (s["scenario_id"], s["flow_type"], t["question_name"]): t
        for s in llm_results
        for t in s["turns"]
    }

    collated_results: dict[tuple, dict] = {}
    relevancy_metric = AnswerRelevancyMetric(
        threshold=threshold, model=eval_model, include_reason=True
    )

    print("\n--- Running Per-Turn Evaluations (this may take a moment) ---")

    for key, gt_turn in gt_lookup.items():
        llm_result_turn = llm_results_lookup.get(key)
        if not llm_result_turn:
            logging.warning(f"No LLM result found for key: {key}")
            continue

        try:
            # Run all three tests for the current turn
            q_consistency_result = evaluate(
                test_cases=[
                    LLMTestCase(
                        input=gt_turn.get("llm_utterance", ""),
                        actual_output=llm_result_turn.get("llm_utterance", ""),
                    )
                ],
                metrics=[relevancy_metric],
            )
            r_appropriateness_result = evaluate(
                test_cases=[
                    LLMTestCase(
                        input=llm_result_turn.get("llm_utterance", ""),
                        actual_output=gt_turn.get("user_utterance", ""),
                    )
                ],
                metrics=[relevancy_metric],
            )

            # Collate results for the current turn
            collated_results[key] = {
                "exact_match_passed": (
                    llm_result_turn.get("actual_output", {})
                    == gt_turn.get("ground_truth_delta", {})
                ),
                "extraction_details": f"Expected: {json.dumps(gt_turn.get('ground_truth_delta', {}))}, Got: {json.dumps(llm_result_turn.get('actual_output', {}))}",
                "question_consistency_score": q_consistency_result.test_results[0]
                .metrics_data[0]
                .score,
                "response_appropriateness_score": r_appropriateness_result.test_results[
                    0
                ]
                .metrics_data[0]
                .score,
            }
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during evaluation for turn {key}: {e}"
            )
    present_results(collated_results)


def present_results(all_results: dict[tuple, dict]):
    """Formats and prints detailed and summary reports to the console."""

    # --- DETAILED TURN-BY-TURN REPORT (now printed first) ---
    print("\n\n" + "=" * 50)
    print("              DETAILED TURN-BY-TURN REPORT")
    print("=" * 50)
    for (scenario_id, flow_type, question_name), res in all_results.items():
        print(f"\n--- Turn: {scenario_id} ({flow_type}) | {question_name} ---")

        if res.get("exact_match_passed"):
            print("  [✅] Extraction Accuracy: PASSED")
        else:
            print("  [❌] Extraction Accuracy: FAILED")
            print("       No LLM result found")

        print(
            f"  [ score: {res['question_consistency_score']:.2f} ] Question Consistency"
        )
        print(
            f"  [ score: {res['response_appropriateness_score']:.2f} ] Response Appropriateness"
        )

    # --- PERFORMANCE SUMMARY REPORT ---
    summary_data = {}
    for (scenario_id, flow_type, question_name), res in all_results.items():
        if res:  # Only include evaluated turns in summary
            summary_data.setdefault(flow_type, []).append(res)

    print("\n\n" + "=" * 50)
    print("              PERFORMANCE SUMMARY REPORT")
    print("=" * 50)
    for flow_type, results_list in summary_data.items():
        if not results_list:
            continue
        total = len(results_list)

        # Calculate averages for each metric
        ea_rate = sum(1 for r in results_list if r["exact_match_passed"]) / total
        qc_avg = sum(r["question_consistency_score"] for r in results_list) / total
        ra_avg = sum(r["response_appropriateness_score"] for r in results_list) / total

        print(f"\n📊 {flow_type.capitalize()} Performance ({total} turns evaluated):")
        print(f"  - Extraction Accuracy: {ea_rate:.2%}")
        print(f"  - Question Consistency Score (Avg): {qc_avg:.2f}")
        print(f"  - Response Appropriateness Score (Avg): {ra_avg:.2f}")
    print("\n" + "=" * 50)


def main():
    """Parses arguments and orchestrates the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run a multi-faceted evaluation on LLM results."
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--gt-file", type=Path, default=script_dir / "data/ground_truth.json"
    )
    parser.add_argument(
        "--results-file", type=Path, default=script_dir / "data/llm_run_results.json"
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

    # Set up API keys
    if not args.openai_key:
        logging.error(
            "OpenAI API key not found. Provide it via --openai-key or set OPENAI_API_KEY."
        )
        return
    os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.deepeval_key:
        os.environ["DEEPEVAL_API_KEY"] = args.deepeval_key

    run_evaluation_suite(
        gt_path=args.gt_file,
        results_path=args.results_file,
        eval_model=args.eval_model,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
