# src/evaluation/evaluator.py

import datetime
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Union # Union for Python < 3.10 for dict | list

from haystack.components.evaluators import (
    AnswerExactMatchEvaluator,
    LLMEvaluator as HaystackLLMEvaluator, # Alias to avoid conflict
    SASEvaluator,
)
from haystack.dataclasses import Document # For type hinting if needed
from langfuse import Langfuse

# --- Project-specific Imports ---
# This section needs to correctly point to your tasks, pipelines, and doc_store
# Adjust based on your actual project structure.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    SRC_ROOT = SCRIPT_DIR.parent
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from ai4gd_momconnect_haystack import doc_store as project_doc_store
    from ai4gd_momconnect_haystack import pipelines
    from ai4gd_momconnect_haystack import tasks
except ImportError as e:
    logging.basicConfig(
        level=logging.INFO
    )  # Ensure logging for this fallback
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.critical(
        "Critical: Failed to import project modules (tasks, etc.): %s. "
        "Ensure evaluator.py is run as part of a package or PYTHONPATH "
        "is set. Example: `python -m src.evaluation.evaluator` from "
        "project root. Using placeholders.",
        e,
    )

    class tasks_placeholder:  # type: ignore
        """Placeholder for tasks module functions."""
        onboarding_flow_id = "onboarding"
        assessment_flow_id = "dma-assessment"

        @staticmethod
        def get_next_onboarding_question(*args, **kwargs):
            """Mock implementation for onboarding question retrieval."""
            return {
                "contextualized_question": "mock_q",
                "collects_field": "mock_f"
            }

        @staticmethod
        def extract_onboarding_data_from_response(r,uc,ch,**kwargs): return uc
        @staticmethod
        def get_assessment_question(*args,**kwargs):
            return {"contextualized_question":"mock_assess_q",
                    "current_question_number":1}
        @staticmethod
        def validate_assessment_answer(*args,**kwargs):
            return {"processed_user_response":"mock_valid_assess_ans"}
    tasks = tasks_placeholder # type: ignore
    class pipelines_placeholder: USE_MOCK_LLM = False # type: ignore
    pipelines = pipelines_placeholder # type: ignore
    class project_doc_store_placeholder: pass # type: ignore
    project_doc_store = project_doc_store_placeholder #type: ignore

from .core_utils import CoreUtils, OutputConfiguration


# --- Constants ---
WHATSAPP_MESSAGE_SIZE_THRESHOLD: int = 400
logger = logging.getLogger(__name__)

# --- Path Configurations ---
BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "src" / "evaluation" / "data"

ONBOARDING_FLOW_DEF_PATH_DEFAULT = EVAL_PATH / "onboarding_flow.json"
ASSESSMENT_FLOW_DEF_PATH_DEFAULT = EVAL_PATH / "assessment_flow.json"
GOLDEN_DATASET_PATH_DEFAULT = EVAL_PATH / "golden_dataset.json"

ONBOARDING_FLOW_DEF_PATH = Path(
    os.getenv("ONBOARDING_FLOW_DEF_PATH", str(ONBOARDING_FLOW_DEF_PATH_DEFAULT))
)
ASSESSMENT_FLOW_DEF_PATH = Path(
    os.getenv("ASSESSMENT_FLOW_DEF_PATH", str(ASSESSMENT_FLOW_DEF_PATH_DEFAULT))
)
GOLDEN_DATASET_PATH = Path(
    os.getenv("GOLDEN_DATASET_PATH", str(GOLDEN_DATASET_PATH_DEFAULT))
)

onboarding_flow_definition_global: dict | None = None
assessment_flow_definition_global: dict | None = None


# --- Helper Functions ---
def load_json_file(file_path: Path) -> dict | list | None:
    """Helper to load a JSON file, returning its content or None on error."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Error: File not found at %s", file_path)
    except json.JSONDecodeError:
        logger.error("Error: Could not decode JSON from %s", file_path)
    except Exception as e:
        logger.error(
            "An unexpected error occurred loading %s: %s", file_path, e
        )
    return None


def get_collects_field_from_question_number(
    q_number: int, flow_def: dict, flow_id_key: str
) -> str | None:
    """Finds 'collects' field for a question number in a flow definition."""
    if not flow_def or not isinstance(flow_def.get(flow_id_key), list):
        logger.warning(
            "Invalid or missing flow definition for flow ID '%s'", flow_id_key
        )
        return None
    for q_data in flow_def[flow_id_key]:
        if q_data.get("question_number") == q_number:
            return q_data.get("collects")
    logger.warning(
        "Could not find 'collects' for Q#%s in flow '%s'.",
        q_number,
        flow_id_key,
    )
    return None


def _get_simulated_response(
    collects_field: str | None,
    simulated_responses_map: dict[str, str],
    scenario_id: str,
) -> str | None:
    """Retrieves a simulated user response from the scenario data."""
    if not collects_field or collects_field not in simulated_responses_map:
        logger.warning(
            "(%s) No simulated response in dataset for field '%s'.",
            scenario_id,
            collects_field,
        )
        return None
    return simulated_responses_map[collects_field]


# --- CORE SIMULATION LOGIC ---
def execute_scenario_simulation(
    dataset_item: dict[str, Any],
    langfuse_instance: Langfuse,
    onboarding_flow_def: dict,
    assessment_flow_def: dict,
) -> tuple[str | None, dict[str, Any], list[str], Any | None]:
    """
    Runs a scenario using tasks.py and pipelines.py, creating a Langfuse trace.

    Returns:
        Tuple of (trace_id, final_user_context, chat_history, trace_object).
    """
    scenario_id = dataset_item.get("scenario_id", "unknown_scenario")
    logger.info(
        "--- Executing Scenario: %s (Pipeline Mocking: %s) ---",
        scenario_id,
        pipelines.USE_MOCK_LLM,
    )

    trace = langfuse_instance.trace(
        name=f"execution-run-{scenario_id}",
        user_id=dataset_item.get("user_persona", {}).get("id", "sim_user"),
        metadata={
            "scenario_input_details": dataset_item,
            "pipeline_mock_status": pipelines.USE_MOCK_LLM,
        },
        tags=[
            dataset_item.get("flow_type", "unknown_flow"),
            "system_execution_run",
        ],
    )

    base_user_context: dict[str, Any] = {
        "age": None, "gender": None, "province": None, "area_type": None,
        "relationship_status": None, "education_level": None,
        "hunger_days": None, "num_children": None, "phone_ownership": None,
        "other": {},
    }
    user_context = {
        **base_user_context,
        **dataset_item.get("user_persona", {}).get(
            "persona_details_from_brief", {}
        ),
    }
    user_context.update(dataset_item.get("initial_user_context", {}))
    chat_history: list[str] = []
    if dataset_item.get("initial_user_utterance"):
        chat_history.append(
            f"User to System: {dataset_item['initial_user_utterance']}"
        )

    simulated_responses = dataset_item.get("simulated_user_responses_map", {})
    final_user_context = user_context.copy()

    if dataset_item["flow_type"] == tasks.onboarding_flow_id:
        logger.info("--- Onboarding Execution for %s ---", scenario_id)
        final_user_context["goal"] = "Complete the onboarding process"
        max_turns = len(simulated_responses) + 3

        for turn_num in range(max_turns):
            span_input = {
                "user_context": final_user_context.copy(),
                "chat_history": chat_history.copy(),
            }
            span = trace.span(
                name=f"onboarding-turn-{turn_num + 1}", input=span_input
            )
            logger.info(
                "(%s) Onboarding Exec Attempt %s", scenario_id, turn_num + 1
            )
            next_q_data = tasks.get_next_onboarding_question(
                final_user_context, chat_history
            )

            if not next_q_data or not isinstance(next_q_data, dict):
                logger.info(
                    "(%s) Onboarding flow ended or task error.", scenario_id
                )
                span.update(
                    output={
                        "status": "Onboarding ended or task error",
                        "task_raw_output": next_q_data,
                    }
                )
                span.end()
                break

            ctx_q = next_q_data.get("contextualized_question")
            coll_f = next_q_data.get("collects_field")
            chosen_q_num = next_q_data.get("chosen_question_number")
            fallback_used = next_q_data.get("fallback_used", False)

            if not coll_f and chosen_q_num is not None:
                coll_f = get_collects_field_from_question_number(
                    chosen_q_num, onboarding_flow_def, tasks.onboarding_flow_id
                )

            span.update(
                metadata={
                    "llm_contextualized_question": ctx_q,
                    "determined_collects_field": coll_f,
                    "chosen_question_number_from_task": chosen_q_num,
                    "task_fallback_used": fallback_used,
                }
            )
            if not ctx_q:
                logger.error(
                    "(%s) No 'contextualized_question' from task. Halting.",
                    scenario_id,
                )
                span.update(
                    output={
                        "status": ("Error: No 'contextualized_question' "
                                   "in task output")
                    }
                )
                span.end()
                break

            chat_history.append(f"System to User: {ctx_q}")
            user_resp = _get_simulated_response(
                coll_f, simulated_responses, scenario_id
            )
            if user_resp is None:
                logger.warning(
                    "(%s) No sim response for field '%s'. Question: '%s'. "
                    "Halting.",
                    scenario_id,
                    coll_f,
                    ctx_q,
                )
                span.update(
                    output={
                        "status": "Halted - no simulated response for field",
                        "field": coll_f,
                        "question_asked": ctx_q,
                    }
                )
                span.end()
                break

            chat_history.append(f"User to System: {user_resp}")
            logger.info(
                "(%s) System: %s || User (for %s): %s",
                scenario_id,
                ctx_q,
                coll_f or "N/A",
                user_resp,
            )

            context_before_extract = final_user_context.copy()
            final_user_context = (
                tasks.extract_onboarding_data_from_response(
                    user_resp,
                    final_user_context,
                    chat_history,
                    expected_collects_field=coll_f,
                )
            )

            extracted_this_turn: dict[str, Any] = {}
            for k, v_after in final_user_context.items():
                v_before = context_before_extract.get(k)
                if k == "other" and isinstance(v_after, dict):
                    other_b = context_before_extract.get("other", {})
                    other_d = {
                        ko: vo
                        for ko, vo in v_after.items()
                        if other_b.get(ko) != vo
                    }
                    if other_d:
                        extracted_this_turn["other"] = other_d
                elif v_before != v_after:
                    extracted_this_turn[k] = v_after

            logger.info(
                "(%s) Data extracted/updated: %s",
                scenario_id,
                json.dumps(extracted_this_turn if extracted_this_turn else "None"),
            )
            span.update(
                output={
                    "user_response": user_resp,
                    "extracted_this_turn": extracted_this_turn,
                }
            )
            span.end()

    elif dataset_item["flow_type"] == tasks.assessment_flow_id:
        logger.info("--- Assessment Execution for %s ---", scenario_id)
        final_user_context["goal"] = "Complete the assessment"
        questions_to_ask = assessment_flow_def.get(tasks.assessment_flow_id, [])

        for step_idx, q_def in enumerate(questions_to_ask):
            coll_f, raw_q_content = q_def["collects"], q_def["content"]
            q_num = q_def["question_number"]
            valid_opts = q_def["valid_responses"]

            if coll_f not in simulated_responses:
                logger.warning(
                    "(%s) No sim response for assess field '%s'. Skipping.",
                    scenario_id,
                    coll_f,
                )
                continue

            span_input = {
                "user_context": final_user_context.copy(),
                "current_assessment_step_index": step_idx,
            }
            span = trace.span(
                name=f"assessment-q-{q_num}",
                input=span_input,
                metadata={
                    "question_number": q_num,
                    "collects_field": coll_f,
                    "raw_question_content": raw_q_content,
                },
            )
            logger.info(
                "(%s) Assessment Exec: Q%s (Idx %s)",
                scenario_id,
                q_num,
                step_idx,
            )

            task_res = tasks.get_assessment_question(
                tasks.assessment_flow_id,
                q_num,
                step_idx,
                final_user_context,
                raw_q_content,
            )
            if not task_res or not task_res.get("contextualized_question"):
                logger.error(
                    "(%s) Failed to get assessment Q%s.", scenario_id, q_num
                )
                span.update(output={"status": "Failed to get question"})
                span.end()
                break

            ctx_q = task_res["contextualized_question"]
            chat_history.append(f"System to User: {ctx_q}")
            user_resp = _get_simulated_response(
                coll_f, simulated_responses, scenario_id
            )
            if user_resp is None:
                logger.error(
                    "(%s) CRITICAL: user_resp None for %s", scenario_id, coll_f
                )
                span.update(output={"status": "ERROR - user_resp is None"})
                span.end()
                break

            chat_history.append(f"User to System: {user_resp}")
            logger.info(
                "(%s) System: %s || User (for %s): %s",
                scenario_id,
                ctx_q,
                coll_f,
                user_resp,
            )

            val_res = tasks.validate_assessment_answer(
                user_resp, q_num, valid_opts
            )
            proc_resp = val_res.get("processed_user_response") if val_res else None
            if proc_resp is not None:
                final_user_context[coll_f] = proc_resp

            span.update(
                output={
                    "user_response": user_resp,
                    "processed_response": proc_resp,
                    "contextualized_question_asked": ctx_q,
                }
            )
            span.update(metadata={"final_user_context_for_turn": final_user_context.copy()})
            span.end()
    else:
        logger.warning(
            "Unknown flow_type: %s for %s",
            dataset_item["flow_type"],
            scenario_id,
        )
        trace.update(
            metadata={"error": f"Unknown flow_type: {dataset_item['flow_type']}"}
        )

    logger.info(
        "--- Execution of Scenario Complete for: %s ---", scenario_id
    )
    trace.update(
        output={
            "final_user_context_after_execution": final_user_context.copy(),
            "final_chat_history_after_execution": chat_history.copy(),
        }
    )
    return trace.id, final_user_context, chat_history, trace


# --- DataPointValidator ---
class DataPointValidator:
    """Validates data points using Haystack or custom Python logic."""

    def __init__(
        self,
        exact_match_evaluator: AnswerExactMatchEvaluator | None = None,
        sas_evaluator: SASEvaluator | None = None,
    ):
        """Initializes validator with optional Haystack evaluators."""
        self.exact_match_evaluator = exact_match_evaluator
        self.sas_evaluator = sas_evaluator
        logger.info(
            "DataPointValidator initialized. ExactMatch: %s, SAS: %s",
            bool(exact_match_evaluator),
            bool(sas_evaluator),
        )

    def _normalize_value(
        self, value: Any, field_name: str | None = None
    ) -> str | None:
        """Normalizes value to lowercase, stripped string; handles num_children."""
        if value is None:
            return None
        val_str = str(value).lower().strip()
        if field_name == "num_children":
            num_map = {
                "one": "1", "two": "2", "three": "3",
                "none": "0", "zero": "0",
            }
            return (
                num_map.get(val_str, val_str)
                if not val_str.isdigit()
                else val_str
            )
        return val_str

    def _get_haystack_score(
        self, result: dict[str, Any] | None, score_key: str
    ) -> float:
        """Extracts a score from a Haystack evaluator result dictionary."""
        if not result:
            return -1.0
        if score_key in result and isinstance(result[score_key], (int, float)):
            return float(result[score_key])
        for res_value in result.values(): # Check nested dicts
            if (isinstance(res_value, dict) and
                score_key in res_value and
                isinstance(res_value[score_key], (int, float))):
                logger.debug(
                    "Found score for '%s' under component in result: %s",
                    score_key, result
                )
                return float(res_value[score_key])
        logger.warning(
            "Could not find score_key '%s' in Haystack eval result: %s",
            score_key, result
        )
        return -1.0

    def validate_single_data_point(
        self,
        collects_field: str,
        llm_extracted_value: Any,
        gt_value: Any,
        valid_responses_list: list[str] | None = None,
        is_required: bool = True,
    ) -> dict[str, Any]:
        """
        Validates a single data point.

        Returns:
            A dictionary with validation flags and values.
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
                    ground_truth_answers=[norm_gt],
                )
                accuracy_score = eval_res.get("score", -1.0)
                is_accurate_to_gt = accuracy_score == 1.0
            elif self.sas_evaluator and collects_field in [
                "your_narrative_field_for_sas_eval"  # Example
            ]:
                eval_res = self.sas_evaluator.run(
                    predicted_answers=[norm_llm],
                    ground_truth_answers=[norm_gt],
                )
                sas_score = self._get_haystack_score(eval_res, "score")
                is_accurate_to_gt = sas_score > 0.8
            else:
                is_accurate_to_gt = norm_llm == norm_gt
        elif norm_llm is None and norm_gt is None:
            is_accurate_to_gt = True

        is_valid_option = False
        if is_collected and norm_llm is not None:
            if valid_responses_list:
                norm_valid_opts = [
                    self._normalize_value(vr, collects_field)
                    for vr in valid_responses_list
                    if vr is not None
                ]
                if self.exact_match_evaluator and norm_valid_opts:
                    for valid_opt_norm in norm_valid_opts:
                        eval_res_opt_check = self.exact_match_evaluator.run(
                            predicted_answers=[norm_llm],
                            ground_truth_answers=[valid_opt_norm],
                        )
                        if eval_res_opt_check.get("score") == 1.0:
                            is_valid_option = True
                            break
                elif norm_valid_opts:
                    is_valid_option = norm_llm in norm_valid_opts
                else:
                    is_valid_option = True
            else:
                is_valid_option = True
        elif not is_collected and (
            norm_gt == "skip"
            or (
                valid_responses_list
                and "skip"
                in [
                    self._normalize_value(vr)
                    for vr in valid_responses_list
                    if vr
                ]
            )
        ):
            is_valid_option = True

        is_req_and_missing = (
            is_required and
            not is_collected and
            not (norm_gt is None or norm_gt == "skip")
        )
        result_dict = {
            "collects_field": collects_field,
            "llm_value_extracted": llm_extracted_value,
            "gt_value": gt_value, "is_collected": is_collected,
            "is_accurate_to_gt": is_accurate_to_gt,
            "is_valid_option": is_valid_option,
            "is_required_and_missing": is_req_and_missing,
        }
        if sas_score != -1.0:
            result_dict["sas_score"] = sas_score
        return result_dict

    def analyze_dialogue_metrics(
        self,
        dialogue_transcript: list[str],
        threshold: int = WHATSAPP_MESSAGE_SIZE_THRESHOLD,
    ) -> dict[str, Any]:
        """Analyzes dialogue for metrics like length and user hesitancy."""
        skips, why_asks, llm_over_thresh, total_llm_msgs = 0, 0, 0, 0
        llm_lengths: list[int] = []
        for entry_str in dialogue_transcript:
            try:
                speaker_token, utterance = entry_str.split(":", 1)
            except ValueError:
                logger.debug("Skipping chat entry due to format: %s", entry_str)
                continue
            utterance = utterance.strip()
            current_speaker = (
                "user" if "User to System" in speaker_token else
                "llm" if "System to User" in speaker_token else
                "unknown"
            )

            if current_speaker == "user":
                utt_low = utterance.lower()
                if "skip" == utt_low: skips += 1
                if ("why do you ask" in utt_low or
                    "why do you need to know" in utt_low): why_asks += 1
            elif current_speaker == "llm":
                total_llm_msgs += 1
                length = len(utterance)
                llm_lengths.append(length)
                if length > threshold: llm_over_thresh += 1

        avg_len = (sum(llm_lengths) / len(llm_lengths)) if llm_lengths else 0.0
        return {
            "total_user_inputs": sum(
                1 for s in dialogue_transcript if s.startswith("User to System:")
            ),
            "total_llm_responses": total_llm_msgs,
            "user_hesitancy": {"skips": skips, "why_asks": why_asks},
            "llm_msg_size": {
                "avg_len": round(avg_len, 2),
                "max_len": max(llm_lengths) if llm_lengths else 0,
                "min_len": min(llm_lengths) if llm_lengths else 0,
                "over_thresh": llm_over_thresh,
                "thresh": threshold,
            },
        }


# --- Evaluator Classes ---
class BaseFlowEvaluator:
    """Base class for flow-specific evaluators."""

    def __init__(
        self,
        flow_definition: list[dict[str, Any]],
        validator: DataPointValidator,
        flow_type_name: str,
        langfuse_client: Langfuse | None = None,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None,
    ):
        """
        Initializes BaseFlowEvaluator.

        Args:
            flow_definition: Question/step definitions for the flow.
            validator: An instance of DataPointValidator.
            flow_type_name: Identifier for the flow type (e.g., "onboarding").
            langfuse_client: Optional Langfuse client for logging scores.
            llm_interaction_evaluator: Optional Haystack LLMEvaluator.
        """
        self.flow_definition = flow_definition
        self.validator = validator
        self.flow_type_name = flow_type_name
        self.field_names = [
            item["collects"] for item in flow_definition if "collects" in item
        ]
        self.llm_interaction_evaluator = llm_interaction_evaluator
        self.langfuse_client = langfuse_client

    def evaluate_completed_simulation(
        self,
        scenario: dict[str, Any],
        execution_trace_id: str | None,
        final_user_context: dict[str, Any],
        chat_history: list[str],
        execution_trace_object: Any | None,
    ) -> dict[str, Any]:
        """Evaluates a scenario post-execution. Must be overridden."""
        raise NotImplementedError(
            "Subclasses must implement evaluate_completed_simulation."
        )

    def _run_qualitative_evaluators(
        self,
        dialogue_transcript: list[str],
        scenario_trace_obj: Any | None
    ) -> dict[str, Any]:
        """Runs qualitative LLM-based evaluators."""
        qual_scores: dict[str, Any] = {
            "eval_status": "qualitative_eval_skipped"
        }
        if not self.llm_interaction_evaluator or not scenario_trace_obj:
            qual_scores["eval_status"] = (
                "llm_interaction_evaluator_or_trace_not_provided"
            )
            return qual_scores

        logger.info(
            "Running qualitative evaluators for trace %s",
            scenario_trace_obj.id
        )
        # Placeholder logic
        if scenario_trace_obj:
            scenario_trace_obj.score(
                name=f"{self.flow_type_name}_avg_llm_clarity_placeholder",
                value=-1.0,
            )
            scenario_trace_obj.score(
                name=f"{self.flow_type_name}_avg_llm_empathy_placeholder",
                value=-1.0,
            )
        qual_scores.update({
            "avg_llm_clarity_placeholder": -1.0,
            "avg_llm_empathy_placeholder": -1.0,
            "eval_status": "qualitative_eval_placeholder_active",
        })
        return qual_scores

    def _aggregate_common_metrics(
        self, reports: list[dict[str, Any]]
    ) -> tuple[dict[str, list[bool]], int, int]:
        """Aggregates common metrics from multiple reports."""
        field_accuracies = {name: [] for name in self.field_names}
        total_msg_over_thresh, total_msg_count = 0, 0
        for report in reports:
            for res_val in report.get("validations", []):
                field = res_val["collects_field"]
                if field in field_accuracies:
                    field_accuracies[field].append(res_val["is_accurate_to_gt"])
            msg_stats = report.get("dialogue_metrics", {}).get("llm_msg_size", {})
            total_msg_over_thresh += msg_stats.get("over_thresh", 0)
            total_msg_count += report.get("dialogue_metrics", {}).get(
                "total_llm_responses", 0
            )
        return field_accuracies, total_msg_over_thresh, total_msg_count

    def _finalize_summary(
        self,
        reports: list[dict[str, Any]],
        specific_metrics: dict[str, Any],
        field_accuracies: dict[str, list[bool]],
        total_msg_over_thresh: int,
        total_msg_count: int,
    ) -> dict[str, Any]:
        """Builds the final summary dictionary for a flow evaluation."""
        field_acc_summary = {
            f: sum(a) / len(a) if a else 0.0
            for f, a in field_accuracies.items()
        }
        perc_over_thresh = (
            (total_msg_over_thresh / total_msg_count * 100)
            if total_msg_count > 0
            else 0.0
        )
        summary_key = f"field_accuracy_rates_{self.flow_type_name}"
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            f"total_{self.flow_type_name}_scenarios": len(reports),
            **specific_metrics,
            summary_key: {
                k: round(v, 3) for k, v in field_acc_summary.items()
            },
            "llm_msg_size_summary": {
                "total_over_thresh": total_msg_over_thresh,
                "perc_over_thresh": round(perc_over_thresh, 2),
                "thresh": WHATSAPP_MESSAGE_SIZE_THRESHOLD,
            },
        }
        return summary

    def run_overall_evaluation(
        self,
        golden_dataset: list[dict[str, Any]],
        output_config: OutputConfiguration,
        langfuse_for_execution: Langfuse,
        onboarding_flow_def: dict,
        assessment_flow_def: dict,
    ) -> dict[str, Any]:
        """Runs execution then evaluation for scenarios and aggregates."""
        reports = []
        specific_metric_values: list[float | int] = []

        for scenario in golden_dataset:
            if scenario.get("flow_type") == self.flow_type_name:
                logger.info(
                    "Executing scenario %s through tasks/pipelines...",
                    scenario["scenario_id"],
                )
                exec_trace_id, final_uc, chat_hist, exec_trace_obj = (
                    execute_scenario_simulation(
                        scenario,
                        langfuse_for_execution,
                        onboarding_flow_def,
                        assessment_flow_def,
                    )
                )
                logger.info(
                    "Execution of %s complete. Trace ID: %s",
                    scenario["scenario_id"],
                    exec_trace_id,
                )

                report = self.evaluate_completed_simulation(
                    scenario,
                    exec_trace_id,
                    final_uc,
                    chat_hist,
                    exec_trace_obj,
                )
                reports.append(report)
                CoreUtils.save_report(
                    report,
                    output_config.detailed_reports_path
                    / f"{report['scenario_id']}.json",
                )

                if self.flow_type_name == "onboarding":
                    specific_metric_values.append(
                        report.get("completeness", {}).get("rate", 0.0)
                    )
                elif self.flow_type_name == "dma-assessment":
                    specific_metric_values.append(
                        1
                        if report.get("order_adherence_simplified", {}).get(
                            "order_ok_for_collected"
                        )
                        else 0
                    )

        field_accuracies, msg_over_thresh, msg_count = (
            self._aggregate_common_metrics(reports)
        )
        avg_specific_metric = (
            (sum(specific_metric_values) / len(specific_metric_values))
            if specific_metric_values
            else 0.0
        )
        specific_summary_metrics: dict[str, Any] = {}

        if self.flow_type_name == "onboarding":
            specific_summary_metrics = {
                "avg_completeness_rate": round(avg_specific_metric, 3),
                "details": [
                    {
                        "id": r["scenario_id"],
                        "comp": r.get("completeness", {}).get("rate"),
                    }
                    for r in reports
                ],
            }
        elif self.flow_type_name == "dma-assessment":
            specific_summary_metrics = {
                "avg_order_adherence_simplified_rate": round(
                    avg_specific_metric, 3
                ),
                "details": [
                    {
                        "id": r["scenario_id"],
                        "order_ok_collected": r.get(
                            "order_adherence_simplified", {}
                        ).get("order_ok_for_collected"),
                    }
                    for r in reports
                ],
            }
            data_acc_scores = [
                1 if r.get("all_data_collected_accurately") else 0
                for r in reports
            ]
            avg_data_acc = (
                sum(data_acc_scores) / len(data_acc_scores)
                if data_acc_scores
                else 0.0
            )
            specific_summary_metrics[
                "avg_assessment_data_accuracy_rate"
            ] = round(avg_data_acc, 3)

        summary = self._finalize_summary(
            reports,
            specific_summary_metrics,
            field_accuracies,
            msg_over_thresh,
            msg_count,
        )
        summary_path = (
            output_config.onboarding_summary_path
            if self.flow_type_name == "onboarding"
            else output_config.assessment_summary_path
        )
        # Adjust key for assessment summary file consistency
        summary_key_to_pop = f"field_accuracy_rates_{self.flow_type_name}"
        if self.flow_type_name == "dma-assessment" and summary_key_to_pop in summary:
            summary[f"field_accuracy_rates_assessment"] = summary.pop(summary_key_to_pop)

        CoreUtils.save_report(summary, summary_path)
        return summary


class OnboardingEvaluator(BaseFlowEvaluator):
    """Evaluates onboarding flow results from actual system execution."""

    def __init__(
        self,
        flow_definition: list[dict[str, Any]],
        validator: DataPointValidator,
        langfuse_client: Langfuse | None = None,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None,
    ):
        """Initializes the OnboardingEvaluator."""
        super().__init__(
            flow_definition,
            validator,
            "onboarding",
            langfuse_client,
            llm_interaction_evaluator,
        )

    def evaluate_completed_simulation(
        self,
        scenario: dict[str, Any],
        execution_trace_id: str | None,
        final_user_context: dict[str, Any],
        chat_history: list[str],
        execution_trace_object: Any | None,
    ) -> dict[str, Any]:
        """Evaluates onboarding results and logs scores to Langfuse trace."""
        scenario_id = scenario.get("scenario_id", f"unknown_{self.flow_type_name}")
        logger.info(
            "Evaluating ONBOARDING results for: %s from trace %s",
            scenario_id,
            execution_trace_id,
        )

        gt_data = scenario.get("ground_truth_extracted_data", {})
        validation_results_detailed: list[dict[str, Any]] = []
        correct_fields_count = 0
        evaluated_field_count = 0

        for item_def in self.flow_definition:
            field = item_def.get("collects")
            if not field:
                continue

            extracted_value = final_user_context.get(field)
            if extracted_value is None and isinstance(
                final_user_context.get("other"), dict
            ):
                extracted_value = final_user_context["other"].get(field)

            is_req = "skip" not in [
                self.validator._normalize_value(vr)
                for vr in item_def.get("valid_responses", [])
                if vr is not None
            ]
            single_validation_result = self.validator.validate_single_data_point(
                field,
                extracted_value,
                gt_data.get(field),
                item_def.get("valid_responses"),
                is_req,
            )
            validation_results_detailed.append(single_validation_result)

            if field in gt_data and self.langfuse_client and execution_trace_object:
                evaluated_field_count += 1
                is_accurate = single_validation_result["is_accurate_to_gt"]
                score_value = 1 if is_accurate else 0
                correct_fields_count += score_value
                score_name = f"onboarding_extraction_accuracy_{field}"
                score_comment = (
                    f"Field: {field}. "
                    f"GT: '{single_validation_result['gt_value']}', "
                    "Actual Extracted: "
                    f"'{single_validation_result['llm_value_extracted']}'. "
                    f"{'ACCURATE' if is_accurate else 'INACCURATE'}"
                )
                try:
                    execution_trace_object.score(
                        name=score_name, value=score_value, comment=score_comment
                    )
                except Exception as e:
                    logger.error("Langfuse score log error for %s: %s", score_name, e)

        if evaluated_field_count > 0 and self.langfuse_client and execution_trace_object:
            overall_accuracy = correct_fields_count / evaluated_field_count
            try:
                execution_trace_object.score(
                    name="onboarding_overall_extraction_accuracy",
                    value=overall_accuracy,
                    comment=(
                        f"Overall: {correct_fields_count}/"
                        f"{evaluated_field_count} fields accurate."
                    ),
                )
            except Exception as e:
                logger.error(
                    "Langfuse score log error for overall_extraction_accuracy: %s", e
                )

        req_total, req_collected_ok = 0, 0
        for res_val in validation_results_detailed:
            field_def = next(
                (f for f in self.flow_definition if f.get("collects") == res_val["collects_field"]), None
            )
            is_req = bool(
                field_def and "skip" not in 
                [self.validator._normalize_value(vr) for vr in field_def.get("valid_responses",[]) if vr]
            )
            if is_req:
                req_total += 1
            if (
                res_val.get("is_collected") and
                res_val.get("is_accurate_to_gt") and
                is_req
            ):  # Check is_req here as well for completeness
                req_collected_ok += 1
        comp_rate = (req_collected_ok / req_total) if req_total > 0 else 1.0

        if self.langfuse_client and execution_trace_object:
            execution_trace_object.score(
                name="onboarding_completeness_rate", value=comp_rate
            )

        dialogue_metrics = self.validator.analyze_dialogue_metrics(chat_history)
        if self.langfuse_client and execution_trace_object:
            for k, v_val in dialogue_metrics.items():
                if isinstance(v_val, dict):
                    for sub_k, sub_v in v_val.items():
                        if sub_k != "thresh" and isinstance(sub_v, (int, float)):
                            execution_trace_object.score(
                                name=f"onboarding_dialogue_{k}_{sub_k}", value=sub_v
                            )
                elif isinstance(v_val, (int, float)):
                    execution_trace_object.score(
                        name=f"onboarding_dialogue_{k}", value=v_val
                    )

        qual_scores = self._run_qualitative_evaluators(
            chat_history, execution_trace_object
        )

        return {
            "scenario_id": scenario_id,
            "flow_type": self.flow_type_name,
            "completeness": {
                "total_required": req_total,
                "collected_correctly": req_collected_ok,
                "rate": round(comp_rate, 3),
                "missing": [
                    r["collects_field"]
                    for r in validation_results_detailed
                    if r.get("is_required_and_missing")
                ],
            },
            "validations": validation_results_detailed,
            "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": chat_history[: min(5, len(chat_history))],
            "notes": scenario.get("notes_for_human_review", ""),
            "execution_trace_id": execution_trace_id if execution_trace_id else "N/A",
        }


class AssessmentEvaluator(BaseFlowEvaluator):
    """Evaluates assessment flow results from actual system execution."""

    def __init__(
        self,
        flow_definition: list[dict[str, Any]],
        validator: DataPointValidator,
        langfuse_client: Langfuse | None = None,
        llm_interaction_evaluator: HaystackLLMEvaluator | None = None,
    ):
        """Initializes the AssessmentEvaluator."""
        super().__init__(
            flow_definition,
            validator,
            "dma-assessment",
            langfuse_client,
            llm_interaction_evaluator,
        )
        self.expected_order = [ # Ensure 'collects' exists
            item["collects"]
            for item in sorted(
                flow_definition, key=lambda x: x.get("question_number", float("inf"))
            )
            if "collects" in item
        ]

    def evaluate_completed_simulation(
        self,
        scenario: dict[str, Any],
        execution_trace_id: str | None,
        final_user_context: dict[str, Any],
        chat_history: list[str],
        execution_trace_object: Any | None,
    ) -> dict[str, Any]:
        """Evaluates assessment results and logs scores to Langfuse trace."""
        scenario_id = scenario.get("scenario_id", f"unknown_{self.flow_type_name}")
        logger.info(
            "Evaluating ASSESSMENT results for: %s from trace %s",
            scenario_id,
            execution_trace_id,
        )

        gt_data = scenario.get("ground_truth_extracted_data", {})
        validation_results_detailed: list[dict[str, Any]] = []

        for item_def in self.flow_definition:
            field = item_def.get("collects")
            if not field:
                continue
            processed_value_from_sim = final_user_context.get(field)
            is_req = "skip" not in [
                self.validator._normalize_value(vr)
                for vr in item_def.get("valid_responses", [])
                if vr is not None
            ]
            single_validation_result = self.validator.validate_single_data_point(
                field,
                processed_value_from_sim,
                gt_data.get(field),
                item_def.get("valid_responses"),
                is_req,
            )
            validation_results_detailed.append(single_validation_result)

            if field in gt_data and self.langfuse_client and execution_trace_object:
                is_accurate = single_validation_result["is_accurate_to_gt"]
                is_canonical = single_validation_result["is_valid_option"]
                try:
                    execution_trace_object.score(
                        name=f"assessment_accuracy_{field}",
                        value=1 if is_accurate else 0,
                        comment=(
                            f"Field: {field}. GT: '{gt_data.get(field)}', "
                            f"Actual: '{processed_value_from_sim}'. "
                            f"ValidOpt: {is_canonical}"
                        ),
                    )
                    execution_trace_object.score(
                        name=f"assessment_is_canonical_option_{field}",
                        value=1 if is_canonical else 0,
                    )
                except Exception as e:
                    logger.error(
                        "Langfuse score log error for %s: %s", field, e
                    )

        llm_collected_fields_in_order = [
            f for f in self.expected_order if f in final_user_context
        ]
        order_ok = llm_collected_fields_in_order == [
            f for f in self.expected_order if f in llm_collected_fields_in_order
        ]
        all_data_ok = all(
            r.get("is_collected") and r.get("is_accurate_to_gt")
            for r in validation_results_detailed
        )

        if self.langfuse_client and execution_trace_object:
            execution_trace_object.score(
                name="assessment_order_adherence_collected", value=1 if order_ok else 0
            )
            execution_trace_object.score(
                name="assessment_all_data_accurate_and_collected",
                value=1 if all_data_ok else 0,
            )

        dialogue_metrics = self.validator.analyze_dialogue_metrics(chat_history)
        if self.langfuse_client and execution_trace_object:
            for k, v_val in dialogue_metrics.items():
                if isinstance(v_val, dict):
                    for sub_k, sub_v in v_val.items():
                        if sub_k != "thresh" and isinstance(sub_v, (int, float)):
                            execution_trace_object.score(
                                name=f"assessment_dialogue_{k}_{sub_k}", value=sub_v
                            )
                elif isinstance(v_val, (int, float)):
                    execution_trace_object.score(
                        name=f"assessment_dialogue_{k}", value=v_val
                    )

        qual_scores = self._run_qualitative_evaluators(
            chat_history, execution_trace_object
        )

        return {
            "scenario_id": scenario_id,
            "flow_type": self.flow_type_name,
            "order_adherence_simplified": {
                "llm_seq_collected": llm_collected_fields_in_order,
                "expected_seq_full": self.expected_order,
                "order_ok_for_collected": order_ok,
            },
            "all_data_collected_accurately": all_data_ok,
            "validations": validation_results_detailed,
            "dialogue_metrics": dialogue_metrics,
            "qualitative_scores": qual_scores,
            "transcript_snippet": chat_history[: min(3, len(chat_history))],
            "execution_trace_id": execution_trace_id if execution_trace_id else "N/A",
        }


# --- Main Runner ---
def main_evaluation_runner(
    golden_dataset_path: Union[str, Path],
    onboarding_flows_json_path: Union[str, Path],
    assessment_flows_json_path: Union[str, Path],
    langfuse_instance: Langfuse | None = None,
    exact_match_evaluator_instance: AnswerExactMatchEvaluator | None = None,
    sas_evaluator_instance: SASEvaluator | None = None,
    llm_interaction_eval_instance: HaystackLLMEvaluator | None = None,
):
    """Main entry point to run evaluations for all configured flows."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format=("%(asctime)s - %(levelname)s - %(name)s - "
                "%(module)s.%(funcName)s:%(lineno)d - %(message)s"), # More detail
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting Main Evaluation Runner (evaluator.py)...")

    global onboarding_flow_definition_global, assessment_flow_definition_global
    onboarding_flow_definition_global, assessment_flow_definition_global = (
        CoreUtils.load_flow_definitions(
            Path(onboarding_flows_json_path), Path(assessment_flows_json_path)
        )
    )

    if not onboarding_flow_definition_global or not onboarding_flow_definition_global.get(tasks.onboarding_flow_id):
        return
    if not assessment_flow_definition_global or not assessment_flow_definition_global.get(tasks.assessment_flow_id):
        logger.critical("Assessment flow data missing for key '%s'. Aborting.", tasks.assessment_flow_id)
        return

    golden_scenarios = CoreUtils.load_golden_dataset(Path(golden_dataset_path))
    if not golden_scenarios:
        logger.critical("No golden scenarios from %s. Aborting.", golden_dataset_path)
        return

    output_conf = OutputConfiguration()
    validator = DataPointValidator(
        exact_match_evaluator_instance, sas_evaluator_instance
    )

    if not langfuse_instance:
        logger.warning(
            "No Langfuse client for main_evaluation_runner; "
            "Langfuse tracing/scoring will be disabled for this run."
        )
        # Create a disabled Langfuse client so methods don't fail
        langfuse_instance = Langfuse(enabled=False)

    logger.info("--- Evaluating Onboarding Flow ---")
    onboarding_eval = OnboardingEvaluator(
        onboarding_flow_definition_global[tasks.onboarding_flow_id],
        validator,
        langfuse_client=langfuse_instance,
        llm_interaction_evaluator=llm_interaction_eval_instance,
    )
    onboarding_summary = onboarding_eval.run_overall_evaluation(
        golden_scenarios,
        output_conf,
        langfuse_instance,  # For execution traces
        onboarding_flow_definition_global,
        assessment_flow_definition_global,
    )
    logger.info("\nOnboarding Evaluation Summary:")
    logger.info(json.dumps(onboarding_summary, indent=2))

    logger.info("\n--- Evaluating Assessment Flow ---")
    assessment_eval = AssessmentEvaluator(
        assessment_flow_definition_global[tasks.assessment_flow_id],
        validator,
        langfuse_client=langfuse_instance,
        llm_interaction_evaluator=llm_interaction_eval_instance,
    )
    assessment_summary = assessment_eval.run_overall_evaluation(
        golden_scenarios,
        output_conf,
        langfuse_instance,  # For execution traces
        onboarding_flow_definition_global,
        assessment_flow_definition_global,
    )
    logger.info("\nAssessment Evaluation Summary:")
    logger.info(json.dumps(assessment_summary, indent=2))

    logger.info(
        "\nEvaluation run complete. Reports saved in: %s",
        output_conf.run_path.resolve(),
    )
    if (
        langfuse_instance and
        getattr(langfuse_instance, 'sdk_integration', None) is not None
    ):  # Check if it's a real client
        langfuse_instance.flush()
        logger.info("Langfuse events flushed.")


if __name__ == "__main__":
    # This block is for testing this evaluator.py script directly.
    current_script_dir = Path(__file__).resolve().parent
    data_dir = current_script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use fixed names for dummy files
    onboarding_key = getattr(tasks, "onboarding_flow_id", "onboarding")
    assessment_key = getattr(tasks, "assessment_flow_id", "dma-assessment")

    dummy_onboarding_fp = data_dir / "dummy_onboarding_flows.json"
    dummy_assessment_fp = data_dir / "dummy_assessment_flows.json"
    dummy_golden_fp = data_dir / "dummy_golden_scenarios.json"

    if not dummy_onboarding_fp.exists():
        CoreUtils.save_report({onboarding_key: [{"question_number": 1, "content": "Province?", "collects": "province", "content_type": "onboarding_message", "valid_responses": ["Gauteng", "Skip"]}]}, dummy_onboarding_fp)
    if not dummy_assessment_fp.exists():
        CoreUtils.save_report({assessment_key: [{"question_number": 1, "content": "Conf Health?", "collects": "conf_health", "content_type": "assessment_question", "valid_responses": ["Confident"]}]}, dummy_assessment_fp)
    if not dummy_golden_fp.exists():
        CoreUtils.save_report([
            {
                "scenario_id": "onboarding_s1", "flow_type": onboarding_key,
                "simulated_user_responses_map": {"province": "I am in Gauteng"},
                "ground_truth_extracted_data": {"province": "Gauteng"}
            },
            {
                "scenario_id": "assessment_s1", "flow_type": assessment_key,
                "simulated_user_responses_map": {"conf_health": "Very confident"},
                "ground_truth_extracted_data": {"conf_health": "Very confident"}
            }
        ], dummy_golden_fp)

    logger_main = logging.getLogger(__name__)  # Ensure logger for __main__
    logger_main.info(
        "Using dummy data from '%s' for __main__ example.",
        data_dir.resolve()
    )

    lf_client = None
    try:
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            lf_client = Langfuse(debug=False)
            if lf_client.auth_check():
                logger_main.info("__main__: Langfuse client OK.")
            else:
                logger_main.error("__main__: Langfuse auth failed.")
                lf_client = None
        else:
            logger_main.warning(
                "__main__: LANGFUSE_PUBLIC_KEY not set. Langfuse disabled."
            )
    except Exception as e:
        logger_main.error(f"__main__: Failed to init Langfuse: {e}.")

    haystack_exact_eval = None
    try:
        haystack_exact_eval = AnswerExactMatchEvaluator()
    except Exception:
        logger_main.warning(
            "__main__: Could not init Haystack AnswerExactMatchEvaluator."
        )

    # For testing the evaluator script with pipeline mocks active:
    os.environ["USE_MOCK_LLM"] = "true"
    os.getenv("RUN_MODE", "EVALUATION")
    logger_main.info(
        "__main__: USE_MOCK_LLM is '%s'", os.getenv("USE_MOCK_LLM")
    )
    # If pipelines module was already imported, its global USE_MOCK_LLM might be stale.
    if 'ai4gd_momconnect_haystack.pipelines' in sys.modules:
        sys.modules['ai4gd_momconnect_haystack.pipelines'].USE_MOCK_LLM = (
            os.getenv("USE_MOCK_LLM", "false").lower() == "true"
        )

    main_evaluation_runner(
        golden_dataset_path=dummy_golden_fp,
        onboarding_flows_json_path=dummy_onboarding_fp,
        assessment_flows_json_path=dummy_assessment_fp,
        langfuse_instance=lf_client,
        exact_match_evaluator_instance=haystack_exact_eval,
    )
