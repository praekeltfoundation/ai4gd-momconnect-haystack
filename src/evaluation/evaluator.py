import json
import datetime
from typing import Any
from pathlib import Path

# --- Required Dependency Imports ---
from langfuse import Langfuse
from langfuse.model import CreateTrace, CreateGeneration, CreateScore

from haystack.evaluation.evaluators import (
    AnswerExactMatchEvaluator, 
    SASEvaluator,
    LLMEvaluator as HaystackLLMEvaluator
)

# --- Local Imports from within the evaluation_suite package ---
from .core_utils import CoreUtils, OutputConfiguration

# --- Constants ---
WHATSAPP_MESSAGE_SIZE_THRESHOLD: int = 400

# --- DialogueSimulator, DataPointValidator, Evaluator Classes ---


class DialogueSimulator:
    """Simulates dialogue interactions and integrates Langfuse tracing."""
    def __init__(
        self, llm_client: Any | None = None,
        langfuse_client: Langfuse | None = None
    ):
        self.llm_client = llm_client
        self.langfuse_client = langfuse_client
        mode = "LIVE LLM (NOT IMPLEMENTED)" if self.llm_client else "MOCK"
        print(f"DialogueSimulator initialized in {mode} mode.")
        if self.langfuse_client:
            print("Langfuse client provided to DialogueSimulator.")
        else:
            print("Warning: Langfuse client not provided. Tracing disabled.")

    def run_simulation(
        self, scenario: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], Any | None]:
        """Runs a dialogue simulation, creating a Langfuse trace."""
        scenario_id: str = scenario.get("scenario_id", "unknown_scenario")
        print(f"\n--- Simulating Scenario: {scenario_id} ---")
        current_trace: Any | None = None

        if self.langfuse_client:
            current_trace = self.langfuse_client.trace(CreateTrace(
                name=f"Eval Scenario - {scenario_id}",
                user_id=scenario.get("user_persona", {}).get("id", "eval_user"),
                metadata={
                    "scenario_id": scenario_id,
                    "flow_type": scenario.get("flow_type")
                },
                tags=[scenario.get("flow_type", "unknown"), "eval_run"]
            ))

        dialogue_transcript: list[dict[str, Any]] = \
            scenario.get("mock_dialogue_transcript", [])
        llm_extracted_data: dict[str, Any] = {}

        if not dialogue_transcript:
            dialogue_transcript = []
            user_inputs = scenario.get("simulated_user_inputs", [])
            for i, user_turn_data in enumerate(user_inputs):
                dialogue_transcript.append(user_turn_data)
                user_utterance = str(user_turn_data.get('utterance', ''))
                ack_text = f"Ack: '{user_utterance[:20]}...'"

                if current_trace:
                    current_trace.generation(CreateGeneration(
                        name=f"LLM Turn {i+1}", input=user_utterance,
                        output=ack_text, model="mock_llm_v1",
                        metadata={"turn_num": i+1}
                    ))
                dialogue_transcript.append({
                    "turn":i+1, "speaker": "llm", "utterance": ack_text,
                    "message_length": len(ack_text)})
        else:
            for i, turn in enumerate(dialogue_transcript):
                if turn.get("speaker") == "llm":
                    if "message_length" not in turn:
                        turn["message_length"] = len(
                            str(turn.get("utterance", ""))
                        )
                    if current_trace:
                        prev_in = (
                            dialogue_transcript[i-1].get("utterance")
                            if i > 0
                            and dialogue_transcript[i-1].get("speaker") == "user"
                            else "N/A"
                        )
                        current_trace.generation(
                            CreateGeneration(
                                name=f"LLM Mocked Turn {turn.get('turn', i)}",
                                input=prev_in, output=turn.get("utterance"),
                                model="mock_transcript_llm",
                                metadata={"turn": turn.get('turn', i)}
                            )
                        )

        llm_extracted_data = scenario.get("mock_llm_extracted_data", {})
        if current_trace:
            current_trace.update(output=llm_extracted_data)

        print(f"--- Simulation End: {scenario_id} ---")
        return dialogue_transcript, llm_extracted_data, current_trace


class DataPointValidator:
    """Validates individual data points using Haystack or custom logic."""
    def __init__(
        self,
        exact_match_evaluator: AnswerExactMatchEvaluator | None = None,
        sas_evaluator: SASEvaluator | None = None
    ):
        self.exact_match_evaluator = exact_match_evaluator
        self.sas_evaluator = sas_evaluator

        init_parts = ["DataPointValidator initialised."]
        if self.exact_match_evaluator:
            init_parts.append("Using Haystack AnswerExactMatchEvaluator.")
        if self.sas_evaluator:
            init_parts.append("Using Haystack SASEvaluator.")
        if not self.exact_match_evaluator and not self.sas_evaluator:
            init_parts.append("Using custom Python logic for matching.")
        print(" ".join(init_parts))

    def _normalize_value(
        self, value: Any, field_name: str | None = None
    ) -> str | None:
        if value is None:
            return None
        val_str = str(value).lower().strip()
        if field_name == "num_children":
            num_map = {
                "one": "1", "two": "2", "three": "3", "none": "0", "zero": "0"
            }
            return (
                num_map.get(val_str, val_str) if not val_str.isdigit()
                else val_str
            )
        return val_str

    def _get_haystack_score(
        self, result: dict[str, Any] | None, score_key: str
    ) -> float:
        """Helper to extract a score from Haystack evaluator result."""
        if not result:
            return -1.0
        if score_key in result:
            val = result[score_key]
            return float(val) if isinstance(val, (int, float)) else -1.0
        first_key = next(iter(result), None)
        if first_key and isinstance(result.get(first_key), dict):
            val = result[first_key].get(score_key, -1.0)
            return float(val) if isinstance(val, (int, float)) else -1.0
        return -1.0

    def validate_single_data_point(
        self, collects_field: str, llm_extracted_value: Any, gt_value: Any,
        valid_responses_list: list[str] | None = None,
        is_required: bool = True
    ) -> dict[str, Any]:
        """
        Validates a single data point against ground truth and valid
        responses.
        """
        norm_llm = self._normalize_value(llm_extracted_value, collects_field)
        norm_gt = self._normalize_value(gt_value, collects_field)
        is_collected = norm_llm is not None
        is_accurate_to_gt = False
        sas_score = -1.0

        if norm_llm is not None and norm_gt is not None:
            if self.exact_match_evaluator:
                eval_res = self.exact_match_evaluator.run(
                    predicted_answers=[norm_llm],
                    ground_truth_answers=[norm_gt]
                )
                is_accurate_to_gt = (
                    self._get_haystack_score(eval_res, "exact_match") == 1.0
                )
            elif self.sas_evaluator and collects_field in ["some_narrative_field"]:  # Example
                eval_res = self.sas_evaluator.run(
                    predicted_answers=[norm_llm], ground_truth_answers=[norm_gt]
                )
                sas_score = self._get_haystack_score(eval_res, "sas_score")
                is_accurate_to_gt = sas_score > 0.8  # Example threshold, to be refined
            else:
                is_accurate_to_gt = (norm_llm == norm_gt)
        elif norm_llm is None and norm_gt is None:
            is_accurate_to_gt = True

        is_valid_option = False
        if is_collected:
            if valid_responses_list:
                norm_valid_responses = [
                    self._normalize_value(vr, collects_field)
                    for vr in valid_responses_list
                ]
                if self.exact_match_evaluator and norm_valid_responses:
                    eval_result = self.exact_match_evaluator.run(
                        predicted_answers=[norm_llm], ground_truth_answers=norm_valid_responses
                    )
                    is_valid_option = self._get_haystack_score(eval_result, "exact_match") == 1.0
                elif norm_valid_responses:
                    is_valid_option = norm_llm in norm_valid_responses
                else:
                    is_valid_option = True
            else:
                is_valid_option = True
        elif (
            norm_gt == "skip"
            or (
                    valid_responses_list
                    and "skip" in [
                        self._normalize_value(vr)
                        for vr in valid_responses_list
                    ]
                )
        ):
            is_valid_option = True
        is_req_and_missing = (
            is_required and not is_collected
            and not (norm_gt is None or norm_gt == "skip")
        )

        result_dict = {
            "collects_field": collects_field,
            "llm_value_extracted": llm_extracted_value,
            "gt_value": gt_value, "is_collected": is_collected,
            "is_accurate_to_gt": is_accurate_to_gt,
            "is_valid_option": is_valid_option,
            "is_required_and_missing": is_req_and_missing
        }
        if sas_score != -1.0:
            result_dict["sas_score"] = sas_score

        return result_dict

    def analyze_dialogue_metrics(
        self, dialogue_transcript: list[dict[str, Any]],
        threshold: int = WHATSAPP_MESSAGE_SIZE_THRESHOLD
    ) -> dict[str, Any]:
        """
        Analyzes dialogue metrics such as user hesitancy and LLM message sizes.
        """
        skips, why_asks, llm_over_thresh, total_llm_msgs = 0, 0, 0, 0
        llm_lengths: list[int] = []
        for turn in dialogue_transcript:
            speaker, utterance = turn.get("speaker"), \
                str(turn.get("utterance", ""))
            if speaker == "user":  # TODO:  Double-check if this is correct
                utt_low = utterance.lower()
                if "skip" == utt_low:
                    skips += 1
                if (
                    "why do you ask" in utt_low
                    or "why do you need to know" in utt_low
                ):
                    why_asks += 1
            elif speaker == "llm":
                total_llm_msgs += 1
                length = turn.get("message_length", len(utterance))
                llm_lengths.append(length)
                if length > threshold:
                    llm_over_thresh += 1
        avg_len = (
            sum(llm_lengths)/len(llm_lengths) if llm_lengths else 0.0
        )
        return {
            "total_user_inputs": sum(
                1 for t in dialogue_transcript if t.get("speaker") == "user"
            ),
            "total_llm_responses": total_llm_msgs,
            "user_hesitancy": {"skips": skips, "why_asks": why_asks},  # TODO: Double-check if this is correct, w.r.t skips
            "llm_msg_size": {
                "avg_len": round(avg_len, 2),
                "max_len": max(llm_lengths) if llm_lengths else 0,
                "min_len": min(llm_lengths) if llm_lengths else 0,
                "over_thresh": llm_over_thresh, "thresh": threshold
            }}


class BaseFlowEvaluator:
    """Base class for flow-specific evaluators."""
    def __init__(
        self,
        flow_definition: list[dict[str, Any]],
        validator: DataPointValidator,
        simulator: DialogueSimulator,
        flow_type_name: str,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None
    ):
        self.flow_definition = flow_definition
        self.validator = validator
        self.simulator = simulator
        self.flow_type_name = flow_type_name
        self.field_names = [item["collects"] for item in flow_definition]
        self.llm_interaction_evaluator = llm_interaction_evaluator

    def evaluate_scenario_base(
        self, scenario: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Any | None]:
        """
        Runs the base evaluation logic for a scenario, simulating dialogue
        """
        dialogue_transcript, llm_extracted_data, scenario_trace = \
            self.simulator.run_simulation(scenario)
        gt_data = scenario.get("ground_truth_extracted_data", {})
        validation_results: list[dict[str, Any]] = []
        for item in self.flow_definition:
            field = item["collects"]
            is_req = "skip" not in [
                self.validator._normalize_value(vr)
                for vr in item.get("valid_responses", [])
            ]
            result = self.validator.validate_single_data_point(
                field, llm_extracted_data.get(field), gt_data.get(field),
                item.get("valid_responses"), is_req
            )
            validation_results.append(result)
        initial_report_data = {
            "dialogue_transcript": dialogue_transcript,
            "llm_extracted_data": llm_extracted_data,
            "gt_data": gt_data
        }
        return initial_report_data, validation_results, scenario_trace

    def _run_qualitative_evaluators(
            self,
            dialogue_transcript: list[dict[str,Any]],
            scenario_trace: Any | None
        ) -> dict[str, Any]:
        """
        Runs qualitative Haystack evaluators (e.g., LLM-based for
        clarity/empathy).
        """
        qual_scores: dict[str, Any] = {
            "eval_status": "qualitative_eval_skipped"
        }
        if not self.llm_interaction_evaluator:
            qual_scores["eval_status"] = \
                "llm_interaction_evaluator_not_provided"
            return qual_scores

        # Placeholder for actual HaystackLLMEvaluator logic
        # TODO: Implement actual LLM evaluation logic
        qual_scores["avg_llm_clarity_placeholder"] = -1.0
        qual_scores["avg_llm_empathy_placeholder"] = -1.0

        if scenario_trace:
            scenario_trace.score(
                    name="avg_llm_clarity",
                    value=qual_scores["avg_llm_clarity_placeholder"]
            )
            scenario_trace.score(
                    name="avg_llm_empathy",
                    value=qual_scores["avg_llm_empathy_placeholder"]
            )
        return qual_scores

    def _aggregate_common_metrics(
        self, reports: list[dict[str, Any]]
    ) -> tuple[dict[str, list[bool]], int, int]:
        """ Aggregates common metrics from evaluation reports. """
        field_accuracies = {name: [] for name in self.field_names}
        total_msg_over_thresh, total_msg_count = 0, 0
        for report in reports:
            for res_val in report["validations"]:
                field = res_val["collects_field"]
                if field in field_accuracies:
                    field_accuracies[field].append(res_val["is_accurate_to_gt"])
            msg_stats = report["dialogue_metrics"]["llm_msg_size"]
            total_msg_over_thresh += msg_stats["over_thresh"]
            total_msg_count += report["dialogue_metrics"]["total_llm_responses"]
        return field_accuracies, total_msg_over_thresh, total_msg_count

    def _finalize_summary(
        self, reports: list[dict[str, Any]],
        specific_metrics: dict[str, Any],
        field_accuracies: dict[str, list[bool]],
        total_msg_over_thresh: int, total_msg_count: int
    ) -> dict[str, Any]:
        field_acc_summary = {
            f: sum(a)/len(a) if a else 0.0 for f, a in field_accuracies.items()
        }
        perc_over_thresh = (
            (total_msg_over_thresh / total_msg_count * 100)
            if total_msg_count > 0
            else 0.0
        )
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            f"total_{self.flow_type_name}_scenarios": len(reports),
            **specific_metrics,
            f"field_accuracy_rates_{self.flow_type_name}": {
                k: round(v, 3) for k, v in field_acc_summary.items()
            },
            "llm_msg_size_summary": {
                "total_over_thresh": total_msg_over_thresh,
                "perc_over_thresh": round(perc_over_thresh, 2),
                "thresh": WHATSAPP_MESSAGE_SIZE_THRESHOLD
            }}
        return summary


class OnboardingEvaluator(BaseFlowEvaluator):
    """Evaluates the onboarding flow data collection."""
    def __init__(
        self, flow_definition: list[dict[str, Any]],
        validator: DataPointValidator, simulator: DialogueSimulator,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None
    ):
        super().__init__(
            flow_definition,
            validator,
            simulator, "onboarding",
            llm_interaction_evaluator=llm_interaction_evaluator
        )

    def evaluate_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Evaluates a single onboarding scenario."""
        scenario_id = scenario.get("scenario_id", f"unknown_{self.flow_type_name}")
        initial_report_data, val_results, scenario_trace = \
            self.evaluate_scenario_base(scenario)
        dialogue_transcript = initial_report_data["dialogue_transcript"]
        req_total, req_collected_ok = 0, 0
        for item, res_val in zip(self.flow_definition, val_results):
            is_req = (
                "skip" not in [
                    self.validator._normalize_value(vr)
                    for vr in item.get("valid_responses", [])
                ]
            )
            if is_req:
                req_total += 1
                if res_val["is_collected"] and res_val["is_accurate_to_gt"]:
                    req_collected_ok += 1
        comp_rate = (req_collected_ok / req_total) if req_total > 0 else 1.0
        dialogue_metrics = \
            self.validator.analyze_dialogue_metrics(dialogue_transcript)
        qual_scores = self._run_qualitative_evaluators(
            dialogue_transcript, scenario_trace
        )
        report = {
            "scenario_id": scenario_id, "flow_type": self.flow_type_name,
            "completeness": {
                "total_required": req_total,
                "collected_correctly": req_collected_ok,
                "rate": round(comp_rate, 3),
                "missing": [
                    r["collects_field"]
                    for r in val_results if r["is_required_and_missing"]
                ]
            },
            "validations": val_results, "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": dialogue_transcript[
                :min(5, len(dialogue_transcript))
            ],
            "notes": scenario.get("notes_for_human_review", "")
        }
        if scenario_trace:
            scenario_trace.score(
                CreateScore(name="onboarding_completeness", value=comp_rate)
            )
            for r_val in val_results:
                scenario_trace.score(CreateScore(
                    name=f"{r_val['collects_field']}_accuracy",
                    value=1 if r_val["is_accurate_to_gt"] else 0,
                    comment=f"LLM: {r_val['llm_value_extracted']}, GT: {r_val['gt_value']}"))
            for k, v_dict in dialogue_metrics.items():
                if isinstance(v_dict, dict):
                    for sub_k, sub_v in v_dict.items():
                        if sub_k != "thresh":
                            scenario_trace.score(
                                CreateScore(name=f"{k}_{sub_k}", value=sub_v)
                            )
                elif isinstance(v_dict, (int, float)):
                    scenario_trace.score(CreateScore(name=k, value=v_dict))
        return report

    def run_evaluation(
        self, golden_dataset: list[dict[str, Any]],
        output_config: OutputConfiguration
    ) -> dict[str, Any]:
        """Runs the evaluation for all onboarding scenarios."""
        reports, completeness_rates = [], []
        for scenario in golden_dataset:
            if scenario.get("flow_type") == self.flow_type_name:
                report = self.evaluate_scenario(scenario)
                reports.append(report)
                CoreUtils.save_report(
                    report,
                    output_config.detailed_reports_path / f"{report['scenario_id']}.json"
                )
                completeness_rates.append(report["completeness"]["rate"])
        field_acc, msg_over_thresh, msg_count = \
            self._aggregate_common_metrics(reports)
        avg_comp_rate = (
            sum(completeness_rates)/len(completeness_rates)
            if completeness_rates else 0.0
        )
        specific_metrics = {
            "avg_completeness_rate": round(avg_comp_rate, 3),
            "details": [
                {"id": r["scenario_id"], "comp": r["completeness"]["rate"]}
                for r in reports
            ]
        }
        summary = self._finalize_summary(
            reports, specific_metrics, field_acc, msg_over_thresh, msg_count
        )
        CoreUtils.save_report(summary, output_config.onboarding_summary_path)
        return summary


class AssessmentEvaluator(BaseFlowEvaluator):
    """Evaluates the assessment flow data collection and order."""
    def __init__(
        self, flow_definition: list[dict[str, Any]], 
        validator: DataPointValidator, simulator: DialogueSimulator,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None
    ):
        super().__init__(
            flow_definition, validator, simulator, "dma-assessment",
            llm_interaction_evaluator=llm_interaction_evaluator
        )
        self.expected_order = [
            item["collects"]
            for item in sorted(
                flow_definition,
                key=lambda x: x.get("question_number", float('inf'))
            )
        ]

    def evaluate_scenario(
        self, scenario: dict[str, Any]
    ) -> dict[str, Any]:
        scenario_id = scenario.get(
            "scenario_id", f"unknown_{self.flow_type_name}"
        )
        initial_report_data, val_results, scenario_trace = \
            self.evaluate_scenario_base(scenario)
        dialogue_transcript = initial_report_data["dialogue_transcript"]
        llm_extracted_data = initial_report_data["llm_extracted_data"]
        llm_reported_seq = scenario.get(
            "mock_llm_collection_sequence",
            list(llm_extracted_data.keys())
        )
        order_devs, order_ok = [], True
        if len(llm_reported_seq) != len(self.expected_order):
            order_ok = False
            order_devs.append(
                f"LLM seq len diff. LLM:{len(llm_reported_seq)},\
                    Exp:{len(self.expected_order)}"
            )
        else:
            for i, expected in enumerate(self.expected_order):
                if (
                    i >= len(llm_reported_seq)
                    or llm_reported_seq[i]
                ) != expected:
                    order_ok = False
                    actual = llm_reported_seq[i] if i < len(llm_reported_seq) \
                        else "None"
                    order_devs.append(
                        f"Pos {i+1}: Exp '{expected}', LLM '{actual}'."
                    )
        all_data_ok = all(
            r["is_collected"] and r["is_accurate_to_gt"] for r in val_results
        )
        final_order_check = order_ok and all_data_ok
        dialogue_metrics = self.validator.analyze_dialogue_metrics(
            dialogue_transcript
        )
        qual_scores = self._run_qualitative_evaluators(
            dialogue_transcript, scenario_trace
        )
        report = {
            "scenario_id": scenario_id,
            "flow_type": self.flow_type_name,
            "order_adherence": {
                "llm_seq": llm_reported_seq,
                "expected_seq": self.expected_order,
                "order_perfect_and_accurate": final_order_check,
                "deviations": order_devs
            },
            "all_data_collected_accurately": all_data_ok,
            "validations": val_results, "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": dialogue_transcript[
                :min(3, len(dialogue_transcript))
            ]
        }
        if scenario_trace:
            scenario_trace.score(
                CreateScore(
                    name="assess_order_perfect_accurate",
                    value=1 if final_order_check else 0
                )
            )
            scenario_trace.score(
                CreateScore(
                    name="assess_all_data_accurate",
                    value=1 if all_data_ok else 0
                    )
                )
            for k, v_dict in dialogue_metrics.items():
                if isinstance(v_dict, dict):
                    for sub_k, sub_v in v_dict.items():
                        if sub_k != "thresh":
                            scenario_trace.score(
                                CreateScore(
                                    name=f"assess_{k}_{sub_k}", value=sub_v
                                )
                            )
                elif isinstance(v_dict, (int, float)):
                    scenario_trace.score(
                        CreateScore(
                            name=f"assess_{k}", value=v_dict
                        )
                    )
        return report

    def run_evaluation(
        self, golden_dataset: list[dict[str, Any]], 
        output_config: OutputConfiguration
    ) -> dict[str, Any]:
        """Runs the evaluation for all assessment scenarios."""
        reports, order_scores, data_comp_acc_scores = [], [], []
        for scenario in golden_dataset:
            if scenario.get("flow_type") == self.flow_type_name:
                report = self.evaluate_scenario(scenario)
                reports.append(report)
                CoreUtils.save_report(
                    report,
                    output_config.detailed_reports_path / f"{report['scenario_id']}.json"
                )
                order_scores.append(
                    1 if report["order_adherence"]["is_order_perfect_and_all_collected_accurately"]
                    else 0
                )
                data_comp_acc_scores.append(
                    1 if report["all_assessment_data_collected_accurately"]
                    else 0
                )
        field_acc, msg_over_thresh, msg_count = self._aggregate_common_metrics(reports)
        avg_order_adherence = (
            sum(order_scores)/len(order_scores)
            if order_scores else 0.0
        )
        avg_data_comp_acc = (
            sum(data_comp_acc_scores)/len(data_comp_acc_scores)
            if data_comp_acc_scores else 0.0
        )
        specific_metrics = {
            "avg_order_adherence_rate": round(avg_order_adherence, 3),
            "avg_data_comp_acc_rate": round(avg_data_comp_acc, 3),
            "details": [
                {
                    "id": r["scenario_id"], "order_ok": r["order_adherence"]["is_order_perfect_and_all_collected_accurately"]
                } for r in reports
            ]
        }
        summary = self._finalize_summary(
            reports, specific_metrics, field_acc, msg_over_thresh, msg_count
        )
        if f"field_accuracy_rates_{self.flow_type_name}" in summary:
            summary["field_accuracy_rates_assessment"] = summary.pop(f"field_accuracy_rates_{self.flow_type_name}")
        CoreUtils.save_report(summary, output_config.assessment_summary_path)
        return summary


def main_evaluation_runner(
    golden_dataset_path: str | Path,
    onboarding_flows_json_path: str | Path,
    assessment_flows_json_path: str | Path,
    langfuse_instance: Langfuse | None = None,
    exact_match_evaluator_instance: AnswerExactMatchEvaluator | None = None,
    sas_evaluator_instance: SASEvaluator | None = None,
    llm_interaction_eval_instance: HaystackLLMEvaluator | None = None
):
    """Main method for running the evaluation suite."""
    print("Starting Main Evaluation Runner...")
    golden_p, onboarding_p, assessment_p = (
        Path(golden_dataset_path),
        Path(onboarding_flows_json_path),
        Path(assessment_flows_json_path)
    )

    output_conf = OutputConfiguration()
    golden_scenarios = CoreUtils.load_golden_dataset(golden_p)
    if not golden_scenarios:
        print(f"Critical: No golden scenarios from {golden_p.resolve()}. Aborting.")
        return

    onboarding_def_dict, assessment_def_dict = CoreUtils.load_flow_definitions(
        onboarding_p, assessment_p
    )
    if (
        not onboarding_def_dict
        or not isinstance(onboarding_def_dict.get("onboarding"), list)
    ):
        print(f"Critical: Onboarding flow error from '{onboarding_p.resolve()}'. Aborting.")
        return
    if (
        not assessment_def_dict
        or not isinstance(assessment_def_dict.get("dma-assessment"), list)
    ):
        print(f"Critical: Assessment flow error from '{assessment_p.resolve()}'. Aborting.")
        return

    simulator = DialogueSimulator(
        llm_client=None, langfuse_client=langfuse_instance
    )  # TODO: Replace None with actual LLM client available
    validator = DataPointValidator(
        exact_match_evaluator=exact_match_evaluator_instance,
        sas_evaluator=sas_evaluator_instance
    )

    print("\n--- Evaluating Onboarding Flow ---")
    onboarding_eval = OnboardingEvaluator(
        onboarding_def_dict["onboarding"], validator, simulator,
        llm_interaction_evaluator=llm_interaction_eval_instance
    )
    onboarding_summary = onboarding_eval.run_evaluation(
        golden_scenarios, output_conf
    )
    print("\nOnboarding Evaluation Summary:")
    print(json.dumps(onboarding_summary, indent=2))

    print("\n--- Evaluating Assessment Flow ---")
    assessment_eval = AssessmentEvaluator(
        assessment_def_dict["dma-assessment"], validator, simulator,
        llm_interaction_evaluator=llm_interaction_eval_instance
    )
    assessment_summary = assessment_eval.run_evaluation(
        golden_scenarios, output_conf
    )
    print("\nAssessment Evaluation Summary:")
    print(json.dumps(assessment_summary, indent=2))

    print("\nEvaluation run complete. Reports saved in:", output_conf.run_path)

    if langfuse_instance:
        langfuse_instance.flush()
        print("Langfuse events flushed.")

# TODO: This if __name__ == "__main__": block is intended for example/testing purposes.
# In a production setup, main_evaluation_runner would be called from a
# dedicated script (e.g., run_evaluation.py in evaluation_suite/).


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dummy_golden_fp = data_dir / "dummy_golden_scenarios.json"
    onboarding_flows_fp = data_dir / "onboarding_flows.json"
    assessment_flows_fp = data_dir / "assessment_flows.json"

    # Create dummy files if they don't exist
    ONBOARDING_DATA = {
        "onboarding": [{"q_num": 1, "collects": "province", "valid_responses": ["Gauteng"," Skip"]}]
    }
    ASSESSMENT_DATA = {
        "dma-assessment": [{"q_num": 1, "collects": "conf_health", "valid_responses":[ "Confident"]}]
    }
    GOLDEN_SCENARIOS_DATA = [
        {
            "scenario_id": "onboarding_s1", "flow_type": "onboarding",
            "mock_dialogue_transcript": [{"speaker": "llm", "utterance": "Province?", "message_length": 10}],
            "ground_truth_extracted_data": {"province": "Gauteng"}, "mock_llm_extracted_data": {"province": "Gauteng"}
        },
        {
            "scenario_id": "assessment_s1", "flow_type": "dma-assessment",
            "mock_dialogue_transcript": [{"speaker": "llm", "utterance": "Conf Health?", "message_length": 12}],
            "ground_truth_extracted_data": {"conf_health": "Confident"}, "mock_llm_extracted_data": {"conf_health": "Confident"},
            "mock_llm_collection_sequence": ["conf_health"]
        }
    ]
    for fp, data_content in [
        (onboarding_flows_fp, ONBOARDING_DATA),
        (assessment_flows_fp, ASSESSMENT_DATA),
        (dummy_golden_fp, GOLDEN_SCENARIOS_DATA)
    ]:
        if not fp.exists():
            CoreUtils.save_report(data_content, fp)
    print(f"\nEnsure dummy files are in '{data_dir.resolve()}'.")

    # --- Initialize Dependencies (CRITICAL: Ensure these are installed and configured) ---
    langfuse_client = None
    try:
        langfuse_client = Langfuse()  # Assumes ENV VARS are set
        print("Langfuse client initialized successfully.")
    except Exception as e:
        print(
            f"CRITICAL: Failed to initialize Langfuse: {e}. "
            "Ensure Langfuse SDK is installed and ENV VARS are set. "
            "Script will run without Langfuse tracing."
        )

    haystack_exact_match_eval, haystack_sas_eval, haystack_llm_q_eval = None, None, None
    try:
        haystack_exact_match_eval = AnswerExactMatchEvaluator()
        print("Haystack AnswerExactMatchEvaluator initialized.")
    except Exception as e:
        print(f"Warning: Failed to init Haystack AnswerExactMatchEvaluator: {e}. "
              "Exact matching will use Python fallback.")
    
    # SASEvaluator and HaystackLLMEvaluator often require specific model/API key setup
    # Initialize them here if you have them configured. Example:
    # try: 
    #     haystack_sas_eval = SASEvaluator(model_name_or_path="your_model_for_sas")
    #     print("Haystack SASEvaluator initialized.")
    # except Exception as e: print(f"Warning: Haystack SAS init error: {e}")
    
    # try: 
    #    haystack_llm_q_eval = HaystackLLMEvaluator(api_key="YOUR_KEY", model_name="gpt-4") 
    #    print("HaystackLLMEvaluator for qualitative checks initialized.")
    # except Exception as e: print(f"Warning: Haystack LLMEvaluator init error: {e}")


    # --- Example Run ---
    # print("\nTo run the main evaluation, uncomment the following lines in the script:")
    # main_evaluation_runner(
    #     golden_dataset_path=dummy_golden_fp,
    #     onboarding_flows_json_path=onboarding_flows_fp,
    #     assessment_flows_json_path=assessment_flows_fp,
    #     langfuse_instance=langfuse_client,
    #     exact_match_evaluator_instance=haystack_exact_match_eval,
    #     sas_evaluator_instance=haystack_sas_eval, 
    #     llm_interaction_eval_instance=haystack_llm_q_eval 
    # )
