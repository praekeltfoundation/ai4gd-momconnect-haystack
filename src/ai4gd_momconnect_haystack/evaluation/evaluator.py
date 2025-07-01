"""
A script to evaluate LLM performance based on three criteria for each turn:
1.  Question Consistency: Is the LLM's question similar to the ground truth question?
2.  User Response Appropriateness: Is the user's answer relevant to the question?
3.  Extraction Accuracy: Does the extracted data exactly match the ground truth?
4.  Intent Classification: Does the classified intent match the ground truth?
"""

import argparse
import asyncio
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric
from deepeval.test_case import LLMTestCase
import polars as pl
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

# This allows the evaluator to call the main simulation script directly.
from ai4gd_momconnect_haystack.main import async_main as run_main_simulation

# Configure logging to show only important messages and suppress third-party warnings
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*")


class Colors:
    """ANSI color codes for console output."""

    OK = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def load_json_file(file_path: Path) -> Optional[List[Dict] | Dict]:
    """Loads and decodes a JSON file."""
    if not file_path.exists():
        logging.error(f"File not found: '{file_path}'.")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from '{file_path}'.")
        return None


def preprocess_text(text: Any) -> str:
    """Normalizes text by lowercasing, removing emojis, and stripping whitespace."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    # Expansive regex to catch a wide range of Unicode emojis and symbols.
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
    text = emoji_pattern.sub(r"", text).strip()
    return re.sub(r"\s+", " ", text)


class Evaluator:
    """Handles the entire evaluation process from data loading to report generation."""

    def __init__(
        self, gt_path: Path, results_path: Path, eval_model: str, threshold: float
    ):
        self.gt_path = gt_path
        self.results_path = results_path
        self.eval_model = eval_model
        self.threshold = threshold
        self.collated_results: Dict[Tuple, Dict] = {}
        self.report_lines: List[str] = []

    def _build_turn_lookup(self, scenarios: List[Dict]) -> Dict[Tuple, Dict]:
        """Creates a lookup map from scenario data for quick access."""
        lookup = {}
        for s in scenarios:
            flow_type = s.get("flow_type", "unknown")
            # The identifier for a turn can vary based on the conversation flow type.
            identifier_key = (
                "question_name"
                if "onboarding" in flow_type or "anc-survey" in flow_type
                else "question_number"
            )
            for t in s.get("turns", []):
                if identifier := t.get(identifier_key):
                    lookup[(s["scenario_id"], flow_type, identifier)] = t
        return lookup

    def _extract_deepeval_result(self, result: Any) -> Tuple[float, Optional[str]]:
        """Safely extracts score and reason from a DeepEval metric result."""
        try:
            metric_data = result.test_results[0].metrics_data[0]
            score = metric_data.score or 0.0
            reason = metric_data.reason
            return score, reason
        except (AttributeError, IndexError):
            return 0.0, "Result extraction failed."

    def run_suite(self):
        """Orchestrates the loading, evaluation, and reporting."""
        print("--- Loading and Collating Evaluation Data ---")
        gt_data = load_json_file(self.gt_path)
        llm_results_data = load_json_file(self.results_path)

        gt_scenarios = (
            gt_data.get("scenarios", []) if isinstance(gt_data, dict) else gt_data
        )
        llm_scenarios = llm_results_data if isinstance(llm_results_data, list) else []

        if not gt_scenarios or not llm_scenarios:
            exit(
                "Evaluation aborted: Could not load or parse ground truth or results data."
            )

        gt_lookup = self._build_turn_lookup(gt_scenarios)
        llm_lookup = self._build_turn_lookup(llm_scenarios)

        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold, model=self.eval_model, include_reason=True
        )

        print("\n--- Running Per-Turn Evaluations (this may take a moment) ---")
        for key, gt_turn in gt_lookup.items():
            if not (llm_turn := llm_lookup.get(key)):
                logging.warning(f"No LLM result found for key: {key}")
                continue

            self._evaluate_turn(key, gt_turn, llm_turn, relevancy_metric)

        self._generate_full_report()

    def _evaluate_turn(
        self, key: Tuple, gt_turn: Dict, llm_turn: Dict, metric: AnswerRelevancyMetric
    ):
        """Runs all evaluations for a single turn and stores the results."""
        try:
            gt_llm_utterance = preprocess_text(gt_turn.get("llm_utterance", ""))
            llm_utterance = preprocess_text(llm_turn.get("llm_utterance", ""))

            # Use the follow-up utterance for appropriateness check if it exists.
            user_utterance = gt_turn.get("follow_up_utterance") or gt_turn.get(
                "user_utterance", ""
            )
            user_utterance_processed = preprocess_text(user_utterance)

            # Explicitly type the metrics list to satisfy mypy's variance rules.
            metrics_list: List[BaseMetric] = [metric]

            # Evaluate semantic similarity of questions and relevance of user's answer.
            q_consistency_result = evaluate(
                test_cases=[
                    LLMTestCase(input=gt_llm_utterance, actual_output=llm_utterance)
                ],
                metrics=metrics_list,
            )
            r_appropriateness_result = evaluate(
                test_cases=[
                    LLMTestCase(
                        input=llm_utterance, actual_output=user_utterance_processed
                    )
                ],
                metrics=metrics_list,
            )

            q_score, q_reason = self._extract_deepeval_result(q_consistency_result)
            r_score, r_reason = self._extract_deepeval_result(r_appropriateness_result)

            self.collated_results[key] = {
                "initial_intent": {
                    "expected": gt_turn.get("intent"),
                    "predicted": llm_turn.get("llm_initial_predicted_intent"),
                },
                "final_intent": {
                    "expected": gt_turn.get("follow_up_intent"),
                    "predicted": llm_turn.get("llm_final_predicted_intent"),
                },
                "extraction": {
                    "expected": preprocess_text(gt_turn.get("user_response", "")),
                    "actual": preprocess_text(llm_turn.get("user_response", "")),
                },
                "question_consistency": {"score": q_score, "reason": q_reason},
                "response_appropriateness": {"score": r_score, "reason": r_reason},
            }

        except Exception as e:
            logging.error(
                f"An unexpected error occurred during evaluation for turn {key}: {e}"
            )

    def _generate_full_report(self, output_path: Optional[Path] = None):
        """Generates and prints all sections of the final report."""
        self._present_turn_by_turn_report()
        self._present_intent_classification_report()
        self._present_performance_summary_report()

        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    # Strip ANSI color codes for the file version
                    clean_report = re.sub(
                        r"\033\[[0-9;]*m", "", "\n".join(self.report_lines)
                    )
                    f.write(clean_report)
                self._add_line(
                    f"\n{Colors.OK}Evaluation report saved to: {output_path}{Colors.ENDC}"
                )
            except IOError as e:
                logging.error(f"Could not write report to file '{output_path}': {e}")

    def _add_line(self, line: str):
        """Helper to print a line to the console and add it to the report buffer."""
        print(line)
        self.report_lines.append(line)

    def _present_turn_by_turn_report(self):
        """Formats and prints the detailed report for each evaluated turn."""
        self._add_line("\n\n" + "=" * 50)
        self._add_line("          DETAILED TURN-BY-TURN REPORT")
        self._add_line("=" * 50)

        for (s_id, flow, t_id), res in self.collated_results.items():
            self._add_line(
                f"\n{Colors.BOLD}--- Turn: {s_id} ({flow}) | ID: {t_id} ---{Colors.ENDC}"
            )

            # Helper to format and print pass/fail status for a given check
            def print_status(name: str, passed: bool, details: str):
                color = Colors.OK if passed else Colors.FAIL
                status = "[✅] PASSED" if passed else "[❌] FAILED"
                self._add_line(f"  {color}{name}: {status}{Colors.ENDC}")
                if not passed:
                    self._add_line(
                        f"       {Colors.FAIL}Details: {details}{Colors.ENDC}"
                    )

            # Intent Checks
            if res["initial_intent"]["expected"]:
                match = (
                    res["initial_intent"]["expected"]
                    == res["initial_intent"]["predicted"]
                )
                details = f"Expected: '{res['initial_intent']['expected']}', Got: '{res['initial_intent']['predicted']}'"
                print_status("Initial Intent", match, details)

            if res["final_intent"]["expected"]:
                match = (
                    res["final_intent"]["expected"] == res["final_intent"]["predicted"]
                )
                details = f"Expected: '{res['final_intent']['expected']}', Got: '{res['final_intent']['predicted']}'"
                print_status("Follow-up Intent", match, details)

            # Extraction Check
            match = res["extraction"]["expected"] == res["extraction"]["actual"]
            details = f"Expected: '{res['extraction']['expected']}', Got: '{res['extraction']['actual']}'"
            print_status("Extraction Accuracy", match, details)

            # Semantic Score Checks
            for key, name in [
                ("question_consistency", "Question Consistency"),
                ("response_appropriateness", "Response Appropriateness"),
            ]:
                score_data = res[key]
                score = score_data["score"]
                reason = score_data["reason"]
                color = Colors.OK if score >= self.threshold else Colors.FAIL
                self._add_line(f"  {color}[ score: {score:.2f} ] {name}{Colors.ENDC}")
                if reason and score < self.threshold:
                    self._add_line(
                        f"       {Colors.WARNING}Reason: {reason}{Colors.ENDC}"
                    )

    def _present_intent_classification_report(self):
        """Generates and prints a classification report and confusion matrix for intents."""
        y_true, y_pred = [], []
        for res in self.collated_results.values():
            for intent_type in ["initial_intent", "final_intent"]:
                if res[intent_type]["expected"]:
                    y_true.append(res[intent_type]["expected"])
                    y_pred.append(res[intent_type]["predicted"])

        if not y_true:
            return

        self._add_line("\n\n" + "=" * 60)
        self._add_line("           INTENT CLASSIFICATION SUMMARY REPORT")
        self._add_line("=" * 60)

        labels = sorted(list(set(y_true + y_pred)))
        class_report = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )
        self._add_line("\n--- Classification Report ---")
        self._add_line(class_report)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pl.DataFrame(cm, schema=[f"Pred: {l}" for l in labels])
        cm_df = cm_df.insert_column(
            0, pl.Series("Actual", [f"Actual: {l}" for l in labels])
        )

        self._add_line("\n--- Confusion Matrix ---")
        self._add_line(str(cm_df))
        self._add_line("=" * 60)

    def _present_performance_summary_report(self):
        """Aggregates results and prints a high-level performance summary."""
        summary_data: Dict[str, List] = {}
        for (s_id, flow_type, q_name), res in self.collated_results.items():
            summary_data.setdefault(flow_type, []).append(res)

        self._add_line("\n\n" + "=" * 50)
        self._add_line("            PERFORMANCE SUMMARY REPORT")
        self._add_line("=" * 50)

        for flow_type, results in summary_data.items():
            total_turns = len(results)
            if total_turns == 0:
                continue

            passed_intent = sum(
                1
                for r in results
                if r["initial_intent"]["expected"]
                and r["initial_intent"]["expected"] == r["initial_intent"]["predicted"]
            )
            total_intent = sum(1 for r in results if r["initial_intent"]["expected"])
            passed_intent += sum(
                1
                for r in results
                if r["final_intent"]["expected"]
                and r["final_intent"]["expected"] == r["final_intent"]["predicted"]
            )
            total_intent += sum(1 for r in results if r["final_intent"]["expected"])

            intent_acc = (passed_intent / total_intent) if total_intent > 0 else 1.0
            extraction_acc = (
                sum(
                    1
                    for r in results
                    if r["extraction"]["expected"] == r["extraction"]["actual"]
                )
                / total_turns
            )
            qc_avg = (
                sum(r["question_consistency"]["score"] for r in results) / total_turns
            )
            ra_avg = (
                sum(r["response_appropriateness"]["score"] for r in results)
                / total_turns
            )

            self._add_line(
                f"\n{Colors.BOLD}📊 {flow_type.capitalize()} Performance ({total_turns} turns):{Colors.ENDC}"
            )
            self._add_line(f"  - Intent Accuracy: {intent_acc:.2%}")
            self._add_line(f"  - Extraction Accuracy: {extraction_acc:.2%}")
            self._add_line(f"  - Avg. Question Consistency: {qc_avg:.2f}")
            self._add_line(f"  - Avg. Response Appropriateness: {ra_avg:.2f}")

        self._add_line("\n" + "=" * 50)


def main():
    """Parses arguments and orchestrates the evaluation."""
    parser = argparse.ArgumentParser(
        description="Run a multi-faceted evaluation on LLM results."
    )
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run simulation to generate a fresh results file.",
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save the report to a text file."
    )
    parser.add_argument(
        "--gt-file", type=Path, default=script_dir / "data/ground_truth.json"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Path to a specific LLM results file to evaluate.",
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
        exit(
            "OpenAI API key not found. Provide it via --openai-key or set OPENAI_API_KEY."
        )
    os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.deepeval_key:
        os.environ["DEEPEVAL_API_KEY"] = args.deepeval_key

    results_path = args.results_file
    if args.run_simulation:
        print("--- Running Automated Simulation ---")
        gt_filename_str = str(args.gt_file)
        result_file_name = None
        if "onboarding" in gt_filename_str:
            result_file_name = "onboarding"
        elif "anc" in gt_filename_str:
            result_file_name = "anc-survey"
        elif "kab" in gt_filename_str:
            result_file_name = "kab"
        elif "dma" in gt_filename_str:
            result_file_name = "dma"

        if result_file_name:
            RESULT_FILE_PATH = (
                script_dir / "data" / f"simulation_results_{result_file_name}.json"
            )
        else:
            RESULT_FILE_PATH = None

        output_path = asyncio.run(
            run_main_simulation(
                GT_FILE_PATH=args.gt_file,
                RESULT_FILE_PATH=RESULT_FILE_PATH,
                is_automated=True,
                save_simulation=True,
            )
        )
        if not output_path or not output_path.exists():
            exit("Simulation failed or did not produce an output file. Aborting.")
        print(f"--- Simulation complete. Using generated file: {output_path} ---")
        results_path = output_path

    if not results_path:
        exit(
            "No results file specified. Use --results-file <path> or add --run-simulation."
        )

    evaluator = Evaluator(
        gt_path=args.gt_file,
        results_path=results_path,
        eval_model=args.eval_model,
        threshold=args.threshold,
    )
    evaluator.run_suite()

    if args.save_report:
        report_path = results_path.with_suffix(".report.txt")
        evaluator._generate_full_report(output_path=report_path)


if __name__ == "__main__":
    main()
