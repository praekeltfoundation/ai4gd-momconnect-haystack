# evaluators.py (or your chosen filename within evaluation_suite)

import json
import datetime
import logging # Added for consistency
from typing import Any, Optional, Union # For Python < 3.10 Union
from pathlib import Path

# --- Required Dependency Imports ---
from langfuse import Langfuse


from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    SASEvaluator,
    LLMEvaluator as HaystackLLMEvaluator
)
# --- Local Imports ---
from .core_utils import CoreUtils, OutputConfiguration


# --- Constants ---
WHATSAPP_MESSAGE_SIZE_THRESHOLD: int = 400
logger = logging.getLogger(__name__)  # Define logger for this module


# --- DialogueSimulator, DataPointValidator, Evaluator Classes ---
class DialogueSimulator:
    """Simulates dialogue interactions and integrates Langfuse tracing."""
    def __init__(
        self, llm_client: Any | None = None,
        langfuse_client: Optional[Langfuse] = None
    ):
        self.llm_client = llm_client # Currently unused in provided logic
        self.langfuse_client = langfuse_client
        mode = "LIVE LLM (NOT IMPLEMENTED)" if self.llm_client else "MOCK (using scenario data)"
        logger.info(f"DialogueSimulator initialized in {mode} mode.")
        if self.langfuse_client:
            logger.info("Langfuse client provided to DialogueSimulator.")
        else:
            logger.warning("Langfuse client not provided. Tracing will be limited or disabled.")

    def run_simulation(
        self, scenario: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], Optional[Any]]: # Optional[Langfuse.Trace]
        """
        Simulates a dialogue based on scenario data, creating a Langfuse trace.
        The "LLM" interactions are based on mock_dialogue_transcript or very simple user_inputs.
        The "extracted data" is taken directly from mock_llm_extracted_data.
        """
        scenario_id: str = scenario.get("scenario_id", "unknown_scenario")
        logger.info(f"\n--- Simulating Scenario: {scenario_id} ---")
        current_trace: Optional[Any] = None

        if self.langfuse_client:
            current_trace = self.langfuse_client.trace(
                name=f"Eval Scenario - {scenario_id}", # More descriptive name
                user_id=scenario.get("user_persona", {}).get("id", "eval_user"),
                metadata={
                    "scenario_id": scenario_id,
                    "description": scenario.get("description", ""),
                    "flow_type": scenario.get("flow_type"),
                    "full_scenario_input": scenario
                },
                tags=[scenario.get("flow_type", "unknown"), "evaluation_suite_run"]
            )

        # Use mock_dialogue_transcript if provided, otherwise simulate simply
        dialogue_transcript: list[dict[str, Any]] = scenario.get("mock_dialogue_transcript", [])

        if not dialogue_transcript and scenario.get("simulated_user_inputs"):
            # This part simulates if no full transcript is given
            # It creates Langfuse generations for these *simulated* LLM turns
            logger.info(f"No mock_dialogue_transcript, generating simple ack from simulated_user_inputs for {scenario_id}")
            temp_transcript: list[dict[str, Any]] = []
            user_inputs = scenario.get("simulated_user_inputs", [])
            turn_counter = 1
            for user_turn_data in user_inputs:
                temp_transcript.append({"turn": turn_counter, **user_turn_data}) # Add user turn
                user_utterance = str(user_turn_data.get('utterance', ''))
                
                # Simulate LLM acknowledgement
                ack_text = f"Acknowledged: '{user_utterance[:30]}...'"
                if current_trace:
                    # This generation is for the *simulator's* action, not a real LLM from tasks.py
                    current_trace.generation(
                        name=f"Simulated LLM Ack Turn {turn_counter}",
                        input={"user_utterance": user_utterance},
                        output={"llm_acknowledgement": ack_text},
                        model="dialogue_simulator_v1_mock",
                        metadata={"turn_num": turn_counter, "scenario_id": scenario_id}
                    )
                temp_transcript.append({
                    "turn": turn_counter, "speaker": "llm", "utterance": ack_text,
                    "message_length": len(ack_text)
                })
                turn_counter += 1
            dialogue_transcript = temp_transcript
        elif dialogue_transcript and current_trace : # Log existing transcript turns
            logger.info(f"Logging existing mock_dialogue_transcript for {scenario_id} to Langfuse.")
            for i, turn_data in enumerate(dialogue_transcript):
                if turn_data.get("speaker") == "llm":
                    # Ensure message length is present
                    if "message_length" not in turn_data:
                        turn_data["message_length"] = len(str(turn_data.get("utterance","")))

                    # Log this pre-defined LLM turn as a generation
                    user_input_for_this_llm_turn = "N/A (start of dialogue)"
                    if i > 0 and dialogue_transcript[i-1].get("speaker") == "user":
                        user_input_for_this_llm_turn = dialogue_transcript[i-1].get("utterance")
                    
                    current_trace.generation(
                        name=f"Transcript LLM Turn {turn_data.get('turn', i+1)}",
                        input={"prior_user_utterance": user_input_for_this_llm_turn},
                        output={"llm_utterance": turn_data.get("utterance")},
                        model="mock_transcript_playback",
                        metadata={"turn_num": turn_data.get('turn', i+1), "scenario_id": scenario_id}
                    )
        
        # This simulator directly uses mock_llm_extracted_data.
        # It does NOT call your tasks.py -> pipelines.py for extraction.
        llm_extracted_data: dict[str, Any] = scenario.get("mock_llm_extracted_data", {})
        logger.info(f"Using mock_llm_extracted_data for {scenario_id}: {llm_extracted_data}")

        if current_trace:
            current_trace.update(output={"simulated_final_extracted_data": llm_extracted_data})

        logger.info(f"--- Simulation End: {scenario_id} ---")
        return dialogue_transcript, llm_extracted_data, current_trace


class DataPointValidator:
    """Validates individual data points using Haystack or custom logic."""
    def __init__(
        self,
        exact_match_evaluator: Optional[AnswerExactMatchEvaluator] = None,
        sas_evaluator: Optional[SASEvaluator] = None
    ):
        self.exact_match_evaluator = exact_match_evaluator
        self.sas_evaluator = sas_evaluator
        # ... (your init print statements) ...
        init_parts = ["DataPointValidator initialised."]
        if self.exact_match_evaluator: init_parts.append("Using Haystack AnswerExactMatchEvaluator.")
        if self.sas_evaluator: init_parts.append("Using Haystack SASEvaluator.")
        if not self.exact_match_evaluator and not self.sas_evaluator: init_parts.append("Using custom Python logic.")
        logger.info(" ".join(init_parts))

    def _normalize_value(self, value: Any, field_name: Optional[str] = None) -> Optional[str]:
        # ... (your existing _normalize_value logic) ...
        if value is None:
            return None
        val_str = str(value).lower().strip()
        if field_name == "num_children": # Example specific normalization
            num_map = {"one": "1", "two": "2", "three": "3", "none": "0", "zero": "0"}
            return num_map.get(val_str, val_str) if not val_str.isdigit() else val_str
        return val_str

    def _get_haystack_score(self, result: Optional[dict[str, Any]], score_key: str) -> float:
        if not result:
            return -1.0
        if score_key in result:
            val = result.get(score_key)
            return float(val) if isinstance(val, (int, float)) else -1.0
        # Check for nested structure
        first_key_in_result = next(iter(result), None)
        if first_key_in_result and isinstance(result.get(first_key_in_result), dict):
            nested_dict = result[first_key_in_result]
            val = nested_dict.get(score_key, -1.0)
            return float(val) if isinstance(val, (int, float)) else -1.0
        return -1.0

    def validate_single_data_point(
        self,
        collects_field: str,
        llm_extracted_value: Any,
        gt_value: Any,
        valid_responses_list: Optional[list[str]] = None,
        is_required: bool = True
    ) -> dict[str, Any]:
        # ... (your existing validation logic to produce the result_dict) ...
        norm_llm = self._normalize_value(llm_extracted_value, collects_field)
        norm_gt = self._normalize_value(gt_value, collects_field)
        is_collected = norm_llm is not None
        is_accurate_to_gt = False
        sas_score = -1.0

        if norm_llm is not None and norm_gt is not None:
            if self.exact_match_evaluator:
                # Haystack evaluators expect list of strings for answers
                eval_res = self.exact_match_evaluator.run(predicted_answers=[norm_llm], ground_truth_answers=[norm_gt])
                print(f"Themba Jonga: is_accurate_to_gt (evals: {eval_res}): {self._get_haystack_score(eval_res, "exact_match")}")
                is_accurate_to_gt = (
                    self._get_haystack_score(eval_res, "score") == 1.0
                )
            elif self.sas_evaluator and collects_field in ["some_narrative_field"]:
                eval_res = self.sas_evaluator.run(predicted_answers=[norm_llm],  ground_truth_answers=[norm_gt])
                sas_score = self._get_haystack_score(eval_res, "score")  # SASEvaluator might use "score"
                is_accurate_to_gt = sas_score > 0.8  # Example threshold
            else:  # Python fallback
                is_accurate_to_gt = (norm_llm == norm_gt)
        elif norm_llm is None and norm_gt is None:  # Both None is considered accurate (e.g., correctly not extracted)
            is_accurate_to_gt = True
        
        is_valid_option = False
        if is_collected:
            if valid_responses_list:
                norm_valid_responses = [self._normalize_value(vr, collects_field) for vr in valid_responses_list if vr is not None]
                if self.exact_match_evaluator and norm_valid_responses:
                    # This use of exact_match_evaluator checks if norm_llm is IN norm_valid_responses
                    # by treating norm_valid_responses as multiple ground truths.
                    # It will be 1.0 if norm_llm matches ANY of them.
                    temp_results_valid_option = []
                    for valid_opt_norm in norm_valid_responses:
                        eval_result_opt_check = self.exact_match_evaluator.run(
                            predicted_answers=[norm_llm],
                            ground_truth_answers=[valid_opt_norm]
                        )
                        if self._get_haystack_score(
                            eval_result_opt_check,
                            "score"
                        ) == 1.0:
                            is_valid_option = True
                            break
                elif norm_valid_responses:
                    is_valid_option = norm_llm in norm_valid_responses
                else:
                    is_valid_option = True
            else:
                is_valid_option = True
        elif not is_collected and (norm_gt == "skip" or (valid_responses_list and "skip" in [self._normalize_value(vr) for vr in valid_responses_list])):
            is_valid_option = True  # If not collected and GT was skip, it's a "valid" outcome for this check

        is_req_and_missing = is_required and not is_collected and not (norm_gt is None or norm_gt == "skip")

        result_dict = {
            "collects_field": collects_field, "llm_value_extracted": llm_extracted_value, "gt_value": gt_value,
            "is_collected": is_collected, "is_accurate_to_gt": is_accurate_to_gt,
            "is_valid_option": is_valid_option, "is_required_and_missing": is_req_and_missing
        }
        if sas_score != -1.0:
            result_dict["sas_score"] = sas_score
        return result_dict

    def analyze_dialogue_metrics(
        self,
        dialogue_transcript: list[dict[str, Any]],
        threshold: int = WHATSAPP_MESSAGE_SIZE_THRESHOLD
    ) -> dict[str, Any]:
        # ... (your existing analyze_dialogue_metrics logic, ensure it handles empty llm_lengths) ...
        skips, why_asks, llm_over_thresh, total_llm_msgs = 0, 0, 0, 0
        llm_lengths: list[int] = []
        for turn in dialogue_transcript:
            speaker, utterance = turn.get("speaker"), str(turn.get("utterance", ""))
            if speaker == "user":
                utt_low = utterance.lower()
                if "skip" == utt_low:
                    skips += 1  # This only counts literal "skip"
                if "why do you ask" in utt_low or "why do you need to know" in utt_low:
                    why_asks += 1
            elif speaker == "llm":
                total_llm_msgs += 1
                length = turn.get("message_length", len(utterance))
                llm_lengths.append(length)
                if length > threshold:
                    llm_over_thresh += 1
        avg_len = (sum(llm_lengths) / len(llm_lengths)) if llm_lengths else 0.0
        return {
            "total_user_inputs": sum(1 for t in dialogue_transcript if t.get("speaker") == "user"),
            "total_llm_responses": total_llm_msgs,
            "user_hesitancy": {"skips": skips, "why_asks": why_asks},
            "llm_msg_size": {
                "avg_len": round(avg_len, 2),
                "max_len": max(llm_lengths) if llm_lengths else 0,
                "min_len": min(llm_lengths) if llm_lengths else 0,
                "over_thresh": llm_over_thresh, "thresh": threshold
            }
        }


class BaseFlowEvaluator:
    """Base class for flow-specific evaluators."""
    def __init__(
        self, flow_definition: list[dict[str, Any]], validator: DataPointValidator,
        simulator: DialogueSimulator, flow_type_name: str,
        llm_interaction_evaluator: Optional[HaystackLLMEvaluator] = None
    ):
        self.flow_definition = flow_definition
        self.validator = validator
        self.simulator = simulator
        self.flow_type_name = flow_type_name
        # Ensure 'collects' exists before trying to access it
        self.field_names = [item["collects"] for item in flow_definition if "collects" in item]
        self.llm_interaction_evaluator = llm_interaction_evaluator

    def evaluate_scenario_base(
        self, scenario: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[Any]]:
        dialogue_transcript, llm_extracted_data, scenario_trace = self.simulator.run_simulation(scenario)
        gt_data = scenario.get("ground_truth_extracted_data", {})
        validation_results: list[dict[str, Any]] = []

        for item_def in self.flow_definition:
            field = item_def.get("collects")
            if not field:  # Skip if question definition doesn't have a "collects" field
                logger.debug(f"Skipping item in flow_definition for scenario {scenario.get('scenario_id')} as it has no 'collects' field: {item_def.get('content')}")
                continue

            is_req = "skip" not in [
                self.validator._normalize_value(vr)
                for vr in item_def.get("valid_responses", []) if vr is not None
            ]
            
            # Access llm_extracted_data, considering it might be flat or have 'other'
            # Your current simulator provides flat mock_llm_extracted_data
            llm_value_for_field = llm_extracted_data.get(field)
            # if llm_value_for_field is None and isinstance(llm_extracted_data.get("other"), dict):
            #     llm_value_for_field = llm_extracted_data["other"].get(field) # If you expect 'other'

            result = self.validator.validate_single_data_point(
                collects_field=field,
                llm_extracted_value=llm_value_for_field,
                gt_value=gt_data.get(field),
                valid_responses_list=item_def.get("valid_responses"),
                is_required=is_req
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
        scenario_trace: Optional[Any]
    ) -> dict[str, Any]:
        # Example: if self.llm_interaction_evaluator and scenario_trace:
        #   eval_result = self.llm_interaction_evaluator.run(transcript=dialogue_transcript)
        #   clarity = eval_result.get("clarity_score", -1.0)
        #   scenario_trace.score(name="llm_clarity_haystack", value=clarity)
        #   return {"llm_clarity_haystack": clarity}
        qual_scores: dict[str, Any] = {"eval_status": "qualitative_eval_skipped"}
        if not self.llm_interaction_evaluator:
            qual_scores["eval_status"] = "llm_interaction_evaluator_not_provided"
            return qual_scores
        # Placeholder for actual HaystackLLMEvaluator logic
        # TODO: Implement actual LLM evaluation logic using dialogue_transcript
        # Example: This would involve formatting the transcript and posing questions to the LLM evaluator
        logger.warning("Placeholder for _run_qualitative_evaluators is active.")
        qual_scores["avg_llm_clarity_placeholder"] = -1.0  # Default if not evaluated
        qual_scores["avg_llm_empathy_placeholder"] = -1.0
        if scenario_trace:
            scenario_trace.score(
                name="avg_llm_clarity_placeholder",
                value=qual_scores["avg_llm_clarity_placeholder"]
            )
            scenario_trace.score(
                name="avg_llm_empathy_placeholder",
                value=qual_scores["avg_llm_empathy_placeholder"]
            )
        return qual_scores

    def _aggregate_common_metrics(self, reports: list[dict[str, Any]]) -> tuple[dict[str, list[bool]], int, int]:
        # ... (your existing _aggregate_common_metrics method) ...
        field_accuracies = {name: [] for name in self.field_names}
        total_msg_over_thresh, total_msg_count = 0, 0
        for report in reports:
            for res_val in report.get("validations", []): # Add .get for safety
                field = res_val["collects_field"]
                if field in field_accuracies: # Ensure field is one we track
                    field_accuracies[field].append(res_val["is_accurate_to_gt"])
            msg_stats = report.get("dialogue_metrics", {}).get("llm_msg_size", {})
            total_msg_over_thresh += msg_stats.get("over_thresh", 0)
            total_msg_count += report.get("dialogue_metrics", {}).get("total_llm_responses", 0)
        return field_accuracies, total_msg_over_thresh, total_msg_count

    def _finalize_summary(
        self, reports: list[dict[str, Any]], specific_metrics: dict[str, Any],
        field_accuracies: dict[str, list[bool]], total_msg_over_thresh: int, total_msg_count: int
    ) -> dict[str, Any]:
        # ... (your existing _finalize_summary method) ...
        field_acc_summary = {f: sum(a)/len(a) if a else 0.0 for f, a in field_accuracies.items()}
        perc_over_thresh = (total_msg_over_thresh / total_msg_count * 100) if total_msg_count > 0 else 0.0
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            f"total_{self.flow_type_name}_scenarios": len(reports),
            **specific_metrics,
            f"field_accuracy_rates_{self.flow_type_name}": {k: round(v, 3) for k, v in field_acc_summary.items()},
            "llm_msg_size_summary": {"total_over_thresh": total_msg_over_thresh, "perc_over_thresh": round(perc_over_thresh, 2), "thresh": WHATSAPP_MESSAGE_SIZE_THRESHOLD}
        }
        return summary


class OnboardingEvaluator(BaseFlowEvaluator):
    """Evaluates the onboarding flow data collection."""
    def __init__(
        self, flow_definition: list[dict[str, Any]], validator: DataPointValidator,
        simulator: DialogueSimulator, llm_interaction_evaluator: Optional[HaystackLLMEvaluator] = None
    ):
        super().__init__(flow_definition, validator, simulator, "onboarding", llm_interaction_evaluator)

    def evaluate_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        scenario_id = scenario.get("scenario_id", f"unknown_{self.flow_type_name}")
        initial_report_data, validation_results, scenario_trace = self.evaluate_scenario_base(scenario)
        
        dialogue_transcript = initial_report_data["dialogue_transcript"]
        gt_data = initial_report_data["gt_data"]  # Ground truth for this scenario

        # --- Langfuse Score Creation for Onboarding Extraction Accuracy ---
        langfuse_scores_to_log = []
        correct_fields_count = 0
        evaluated_field_count = 0

        for val_res in validation_results:  # Iterate over results from DataPointValidator
            field = val_res["collects_field"]
            if field in gt_data:  # Only score fields that have a ground truth for this scenario
                evaluated_field_count += 1
                is_accurate = val_res["is_accurate_to_gt"]
                score_value = 1 if is_accurate else 0
                correct_fields_count += score_value
                
                langfuse_scores_to_log.append(self.simulator.langfuse_client.score(
                    trace_id=scenario_trace.id if scenario_trace else None,
                    name=f"onboarding_extraction_accuracy_{field}",
                    value=score_value,
                    comment=(f"Field: {field}. Expected: '{val_res['gt_value']}', "
                             f"LLM Got: '{val_res['llm_value_extracted']}'. "
                             f"{'ACCURATE' if is_accurate else 'INACCURATE'}")
                ))
        
        if evaluated_field_count > 0:
            overall_accuracy = correct_fields_count / evaluated_field_count
            langfuse_scores_to_log.append(self.simulator.langfuse_client.score(
                trace_id=scenario_trace.id if scenario_trace else None,
                name="onboarding_overall_extraction_accuracy", # Langfuse score name
                value=overall_accuracy,
                comment=f"Overall: Matched {correct_fields_count}/{evaluated_field_count} expected fields."
            ))

        # Log these scores to Langfuse via the simulator's client
        if self.simulator.langfuse_client and scenario_trace:
            for score_obj in langfuse_scores_to_log:
                print(f"Langfuse with score_obj {score_obj}")
                if score_obj.trace_id:  # Ensure trace_id is set
                    try:
                        self.simulator.langfuse_client.score(score_obj)
                    except Exception as e_score:
                        pass
                        # logger.error(f"Failed to log score {score_obj.name} to Langfuse: {e_score}")
        # --- End of Langfuse Score Creation ---

        # Your existing completeness calculation
        req_total, req_collected_ok = 0, 0
        for item_def, res_val in zip(self.flow_definition, validation_results): # Assuming validation_results align with flow_definition
            is_req = "skip" not in [self.validator._normalize_value(vr) for vr in item_def.get("valid_responses", []) if vr is not None]
            if is_req:
                req_total += 1
                if res_val["is_collected"] and res_val["is_accurate_to_gt"]:
                    req_collected_ok += 1
        comp_rate = (req_collected_ok / req_total) if req_total > 0 else 1.0
        
        dialogue_metrics = self.validator.analyze_dialogue_metrics(dialogue_transcript)
        qual_scores = self._run_qualitative_evaluators(dialogue_transcript, scenario_trace)
        
        report = {
            "scenario_id": scenario_id, "flow_type": self.flow_type_name,
            "completeness": {"total_required": req_total, "collected_correctly": req_collected_ok, "rate": round(comp_rate, 3),
                             "missing": [r["collects_field"] for r in validation_results if r["is_required_and_missing"]]},
            "validations": validation_results, "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": dialogue_transcript[:min(5, len(dialogue_transcript))],
            "notes": scenario.get("notes_for_human_review", "")
        }
        
        # Your existing Langfuse score logging for high-level metrics
        if scenario_trace:
            scenario_trace.score(name="onboarding_completeness_metric", value=comp_rate) # Renamed to avoid clash
            # Dialogue metrics logging
            for k, v_dict in dialogue_metrics.items():
                if isinstance(v_dict, dict):
                    for sub_k, sub_v in v_dict.items():
                        if sub_k != "thresh" and isinstance(sub_v, (int, float)):
                            scenario_trace.score(name=f"onboarding_{k}_{sub_k}", value=sub_v)
                elif isinstance(v_dict, (int, float)):
                    scenario_trace.score(name=f"onboarding_{k}", value=v_dict)
        return report

    def run_evaluation(self, golden_dataset: list[dict[str, Any]], output_config: OutputConfiguration) -> dict[str, Any]:
        # ... (your existing run_evaluation, ensure it calls the updated evaluate_scenario) ...
        reports, completeness_rates = [], []
        for scenario in golden_dataset:
            if scenario.get("flow_type") == self.flow_type_name:
                report = self.evaluate_scenario(scenario)
                reports.append(report)
                CoreUtils.save_report(report, output_config.detailed_reports_path / f"{report['scenario_id']}.json")
                completeness_rates.append(report["completeness"]["rate"])
        
        field_accuracies, msg_over_thresh, msg_count = self._aggregate_common_metrics(reports)
        avg_comp_rate = (sum(completeness_rates) / len(completeness_rates)) if completeness_rates else 0.0
        specific_metrics = {"avg_completeness_rate": round(avg_comp_rate, 3),
                            "details": [{"id": r["scenario_id"], "comp": r["completeness"]["rate"]} for r in reports]}
        summary = self._finalize_summary(reports, specific_metrics, field_accuracies, msg_over_thresh, msg_count)
        CoreUtils.save_report(summary, output_config.onboarding_summary_path)
        return summary


class AssessmentEvaluator(BaseFlowEvaluator):
    """Evaluates the assessment flow data collection and order."""
    def __init__(
        self, flow_definition: list[dict[str, Any]], validator: DataPointValidator,
        simulator: DialogueSimulator, llm_interaction_evaluator: Optional[HaystackLLMEvaluator] = None
    ):
        super().__init__(flow_definition, validator, simulator, "dma-assessment", llm_interaction_evaluator)
        self.expected_order = [item["collects"] for item in sorted(flow_definition, key=lambda x: x.get("question_number", float('inf'))) if "collects" in item]

    def evaluate_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        scenario_id = scenario.get("scenario_id", f"unknown_{self.flow_type_name}")
        initial_report_data, validation_results, scenario_trace = self.evaluate_scenario_base(scenario)
        
        dialogue_transcript = initial_report_data["dialogue_transcript"]
        llm_extracted_data = initial_report_data["llm_extracted_data"] # This is mock_llm_extracted_data
        gt_data = initial_report_data["gt_data"]

        # --- Langfuse Score Creation for Assessment Validation Accuracy ---
        langfuse_scores_to_log = []

        for val_res in validation_results:  # validation_results are per-field from flow_definition
            field = val_res["collects_field"]
            # Create scores similar to score_assessment_validation_and_processing
            # Here, llm_extracted_data directly contains the "processed" value from the mock
            # The DataPointValidator's is_accurate_to_gt checks this against gt_data.
            # is_valid_option checks if it's one of the enum values.
            
            # Score 1: Accuracy against ground truth for this assessment field
            is_accurate = val_res["is_accurate_to_gt"]
            langfuse_scores_to_log.append(self.simulator.langfuse_client.score(
                trace_id=scenario_trace.id if scenario_trace else None,
                name=f"assessment_accuracy_{field}", # Langfuse score name
                value=1 if is_accurate else 0,
                comment=(f"Field: {field}. Expected: '{val_res['gt_value']}', "
                         f"LLM (Mock) Processed: '{val_res['llm_value_extracted']}'. "
                         f"{'ACCURATE' if is_accurate else 'INACCURATE'}")
            ))

            # Score 2: Was the (mocked) LLM processed response a canonical option?
            is_canonical = val_res["is_valid_option"]
            field_def = next((f_def for f_def in self.flow_definition if f_def.get("collects") == field), None)
            valid_options_for_q = field_def.get("valid_responses", []) if field_def else []
            langfuse_scores_to_log.append(self.simulator.langfuse_client.score(
                trace_id=scenario_trace.id if scenario_trace else None,
                name=f"assessment_is_canonical_option_{field}",
                value=1 if is_canonical else 0,
                comment=(f"Field: {field}. LLM (Mock) Processed: '{val_res['llm_value_extracted']}'. "
                         f"Is Canonical: {is_canonical}. Valid options: {valid_options_for_q}")
            ))

        # Log these scores to Langfuse
        if self.simulator.langfuse_client and scenario_trace:
            for score_obj in langfuse_scores_to_log:
                if score_obj.trace_id:
                    try:
                        self.simulator.langfuse_client.score(score_obj)
                    except Exception as e_score:
                        pass
                        # logger.error(f"Failed to log assessment score {score_obj.name} to Langfuse: {e_score}")
        # --- End of Langfuse Score Creation ---

        # Your existing order adherence logic
        llm_reported_seq = scenario.get("mock_llm_collection_sequence", list(llm_extracted_data.keys()))
        order_devs, order_ok = [], True
        # ... (rest of your order adherence logic from the original script) ...
        if len(llm_reported_seq) != len(self.expected_order):
            order_ok = False
            order_devs.append(f"LLM seq len diff. LLM:{len(llm_reported_seq)}, Exp:{len(self.expected_order)}")
        else:
            for i, expected_field in enumerate(self.expected_order):
                if i >= len(llm_reported_seq) or llm_reported_seq[i] != expected_field:
                    order_ok = False
                    actual = llm_reported_seq[i] if i < len(llm_reported_seq) else "None"
                    order_devs.append(f"Pos {i+1}: Exp '{expected_field}', LLM '{actual}'.")

        all_data_ok = all(r["is_collected"] and r["is_accurate_to_gt"] for r in validation_results)
        final_order_check = order_ok and all_data_ok
        
        dialogue_metrics = self.validator.analyze_dialogue_metrics(dialogue_transcript)
        qual_scores = self._run_qualitative_evaluators(dialogue_transcript, scenario_trace)
        
        report = { # Your existing report structure
            "scenario_id": scenario_id, "flow_type": self.flow_type_name,
            "order_adherence": {"llm_seq": llm_reported_seq, "expected_seq": self.expected_order,
                                "order_perfect_and_accurate": final_order_check, "deviations": order_devs},
            "all_data_collected_accurately": all_data_ok,
            "validations": validation_results, "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": dialogue_transcript[:min(3, len(dialogue_transcript))]
        }
        
        # Your existing high-level Langfuse scores
        if scenario_trace:
            scenario_trace.score(name="assessment_order_perfect_and_accurate", value=1 if final_order_check else 0)
            scenario_trace.score(name="assessment_all_data_collected_accurately", value=1 if all_data_ok else 0)
            # Dialogue metrics logging
            for k, v_dict in dialogue_metrics.items():
                if isinstance(v_dict, dict):
                    for sub_k, sub_v in v_dict.items():
                        if sub_k != "thresh" and isinstance(sub_v, (int, float)):
                            scenario_trace.score(name=f"assessment_{k}_{sub_k}", value=sub_v)
                elif isinstance(v_dict, (int, float)):
                    scenario_trace.score(name=f"assessment_{k}", value=v_dict)
        return report

    def run_evaluation(self, golden_dataset: list[dict[str, Any]], output_config: OutputConfiguration) -> dict[str, Any]:
        # ... (your existing run_evaluation, ensuring it calls the updated evaluate_scenario) ...
        reports, order_scores, data_comp_acc_scores = [], [], [] # Renamed for clarity
        for scenario in golden_dataset:
            if scenario.get("flow_type") == self.flow_type_name:
                report = self.evaluate_scenario(scenario)
                reports.append(report)
                CoreUtils.save_report(report, output_config.detailed_reports_path / f"{report['scenario_id']}.json")
                # Use more descriptive keys from the report for clarity
                order_scores.append(1 if report["order_adherence"]["order_perfect_and_accurate"] else 0)
                data_comp_acc_scores.append(1 if report["all_data_collected_accurately"] else 0)
        
        field_accuracies, msg_over_thresh, msg_count = self._aggregate_common_metrics(reports)
        avg_order_adherence = (sum(order_scores) / len(order_scores)) if order_scores else 0.0
        avg_data_comp_acc = (sum(data_comp_acc_scores) / len(data_comp_acc_scores)) if data_comp_acc_scores else 0.0
        specific_metrics = {
            "avg_order_adherence_rate": round(avg_order_adherence, 3),
            "avg_data_collection_accuracy_rate": round(avg_data_comp_acc, 3), # Renamed for clarity
            "details": [{"id": r["scenario_id"], "order_ok_acc": r["order_adherence"]["order_perfect_and_accurate"]} for r in reports]
        }
        summary = self._finalize_summary(reports, specific_metrics, field_accuracies, msg_over_thresh, msg_count)
        # Ensure correct summary key for assessment
        if f"field_accuracy_rates_{self.flow_type_name}" in summary:
            summary[f"field_accuracy_rates_assessment"] = summary.pop(f"field_accuracy_rates_{self.flow_type_name}")
        CoreUtils.save_report(summary, output_config.assessment_summary_path)
        return summary


def main_evaluation_runner(
    golden_dataset_path: Union[str, Path], # Use Union for older Python if | not allowed
    onboarding_flows_json_path: Union[str, Path],
    assessment_flows_json_path: Union[str, Path],
    langfuse_instance: Optional[Langfuse] = None,
    exact_match_evaluator_instance: Optional[AnswerExactMatchEvaluator] = None,
    sas_evaluator_instance: Optional[SASEvaluator] = None,
    llm_interaction_eval_instance: Optional[HaystackLLMEvaluator] = None
):
    # ... (your existing main_evaluation_runner logic, which seems fine) ...
    # Ensure logger is configured here or globally if used before basicConfig
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.info("Starting Main Evaluation Runner...") # Use logger

    golden_p, onboarding_p, assessment_p = Path(golden_dataset_path), Path(onboarding_flows_json_path), Path(assessment_flows_json_path)
    output_conf = OutputConfiguration() # Ensure this is correctly initialized

    golden_scenarios = CoreUtils.load_golden_dataset(golden_p)
    if not golden_scenarios:
        logger.critical(f"Critical: No golden scenarios from {golden_p.resolve()}. Aborting.")
        return

    onboarding_def_dict, assessment_def_dict = CoreUtils.load_flow_definitions(onboarding_p, assessment_p)
    if not onboarding_def_dict or not isinstance(onboarding_def_dict.get("onboarding"), list) or not onboarding_def_dict.get("onboarding"):
        logger.critical(f"Critical: Onboarding flow error or empty from '{onboarding_p.resolve()}'. Aborting.")
        return
    if not assessment_def_dict or not isinstance(assessment_def_dict.get("dma-assessment"), list) or not assessment_def_dict.get("dma-assessment"):
        logger.critical(f"Critical: Assessment flow error or empty from '{assessment_p.resolve()}'. Aborting.")
        return

    simulator = DialogueSimulator(llm_client=None, langfuse_client=langfuse_instance)
    validator = DataPointValidator(exact_match_evaluator=exact_match_evaluator_instance, sas_evaluator=sas_evaluator_instance)

    logger.info("\n--- Evaluating Onboarding Flow ---")
    onboarding_eval = OnboardingEvaluator(
        onboarding_def_dict["onboarding"], validator, simulator,
        llm_interaction_evaluator=llm_interaction_eval_instance
    )
    onboarding_summary = onboarding_eval.run_evaluation(golden_scenarios, output_conf)
    logger.info("\nOnboarding Evaluation Summary:")
    logger.info(json.dumps(onboarding_summary, indent=2))

    logger.info("\n--- Evaluating Assessment Flow ---")
    assessment_eval = AssessmentEvaluator(
        assessment_def_dict["dma-assessment"], validator, simulator,
        llm_interaction_evaluator=llm_interaction_eval_instance
    )
    assessment_summary = assessment_eval.run_evaluation(golden_scenarios, output_conf)
    logger.info("\nAssessment Evaluation Summary:")
    logger.info(json.dumps(assessment_summary, indent=2))

    logger.info(f"\nEvaluation run complete. Reports saved in: {output_conf.run_path.resolve()}")

    if langfuse_instance:
        langfuse_instance.flush()
        logger.info("Langfuse events flushed.")


if __name__ == "__main__":
    # Simplified __main__ for clarity, assuming files are in a 'data' subdir of this script's dir
    current_script_dir = Path(__file__).resolve().parent
    # This assumes your data files (golden, onboarding_flows, assessment_flows)
    # are in a directory structure that this example can reach.
    # For instance, if 'data' is a sibling to 'evaluation_suite' package,
    # and this script is inside 'evaluation_suite'.
    # Adjust 'data_dir' path as per your actual project structure.
    # Example: data_dir = current_script_dir.parent / "data" # if data is one level up from this script's dir
    
    # For the dummy data generation in your example, let's assume a 'data' subdir
    # in the same directory as this evaluators.py for the dummy files.
    # If evaluators.py is at project_root/evaluation_suite/evaluators.py
    # then data_dir would be project_root/evaluation_suite/data/
    data_dir = current_script_dir / "data" 
    data_dir.mkdir(parents=True, exist_ok=True) # Ensure data dir exists for dummy files

    dummy_golden_fp = data_dir / "dummy_golden_scenarios.json"
    onboarding_flows_fp = data_dir / "onboarding_flows.json" # Should point to your actual file
    assessment_flows_fp = data_dir / "assessment_flows.json" # Should point to your actual file

    # Create dummy files only if they don't exist (as per your example)
    # In a real run, these paths should point to your actual JSON definition files.
    # For the dummy data, ensure CoreUtils.save_report is functional or use simple json.dump
    if not onboarding_flows_fp.exists():
        ONBOARDING_DATA = {"onboarding": [{"question_number": 1, "content": "Province?", "collects": "province", "content_type": "onboarding_message", "valid_responses": ["Gauteng"," Skip"]}]}
        CoreUtils.save_report(ONBOARDING_DATA, onboarding_flows_fp)
    if not assessment_flows_fp.exists():
        ASSESSMENT_DATA = {"dma-assessment": [{"question_number": 1, "content":"Conf Health?", "collects": "conf_health", "content_type":"assessment_question", "valid_responses":[ "Confident"]}]}
        CoreUtils.save_report(ASSESSMENT_DATA, assessment_flows_fp)
    if not dummy_golden_fp.exists():
        GOLDEN_SCENARIOS_DATA = [
            {"scenario_id": "onboarding_s1", "flow_type": "onboarding", "mock_dialogue_transcript": [{"speaker": "llm", "utterance": "Province?", "message_length": 10}], "ground_truth_extracted_data": {"province": "Gauteng"}, "mock_llm_extracted_data": {"province": "Gauteng"}},
            {"scenario_id": "assessment_s1", "flow_type": "dma-assessment", "mock_dialogue_transcript": [{"speaker": "llm", "utterance": "Conf Health?", "message_length": 12}], "ground_truth_extracted_data": {"conf_health": "Confident"}, "mock_llm_extracted_data": {"conf_health": "Confident"}, "mock_llm_collection_sequence": ["conf_health"]}
        ]
        CoreUtils.save_report(GOLDEN_SCENARIOS_DATA, dummy_golden_fp)
    
    logger.info(f"Using dummy/example data files from '{data_dir.resolve()}' if actual files not found at specified paths.")

    langfuse_client = None
    try:
        # Ensure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST are set in your environment
        langfuse_client = Langfuse()
        if langfuse_client.auth_check():
            logger.info("Langfuse client initialized and authenticated successfully.")
        else:
            logger.error("Langfuse authentication failed. Tracing will be disabled.")
            langfuse_client = None # Disable if auth fails
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}. Tracing disabled. Ensure SDK installed & ENV VARS set.")

    haystack_exact_match_eval = None
    try:
        haystack_exact_match_eval = AnswerExactMatchEvaluator()
        logger.info("Haystack AnswerExactMatchEvaluator initialized.")
    except Exception as e:
        logger.warning(f"Failed to init Haystack AnswerExactMatchEvaluator: {e}. Exact matching will use Python fallback.")
    
    # Initialize other Haystack evaluators if configured and needed
    haystack_sas_eval = None
    haystack_llm_q_eval = None

    print(f'Dummy golden scenarios file path: {dummy_golden_fp.resolve()}')
    main_evaluation_runner(
        golden_dataset_path=dummy_golden_fp,
        onboarding_flows_json_path=onboarding_flows_fp, # Replace with actual
        assessment_flows_json_path=assessment_flows_fp, # Replace with actual
        langfuse_instance=langfuse_client,
        exact_match_evaluator_instance=haystack_exact_match_eval,
        sas_evaluator_instance=haystack_sas_eval,
        llm_interaction_eval_instance=haystack_llm_q_eval
    )
