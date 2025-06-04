# main.py (or your evaluation_runner.py)
import json
import logging
import os
import sys
import traceback
from typing import Any, Optional, Union # Union for Python < 3.10 for dict | list
from pathlib import Path

from langfuse import Langfuse

# Adjust this import based on your project structure
# If main.py is at project root, and tasks.py is in src/ai4gd_momconnect_haystack/tasks.py:
# SCRIPT_DIR = Path(__file__).resolve().parent
# SRC_DIR = SCRIPT_DIR / "src" # Or wherever your package lives relative to this script
# sys.path.insert(0, str(SRC_DIR.parent)) # Add the directory *containing* your package
# from ai4gd_momconnect_haystack import tasks
# For now, using relative import assuming it's part of the package
from . import tasks
from . import doc_store


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
# Example: if your data files are in 'project_root/evaluation/data/'
# and this script is in 'project_root/src/app/'
# PROJECT_ROOT = BASE_DIR.parent.parent # Goes up two levels from 'app' to 'project_root'
# EVAL_PATH = PROJECT_ROOT / "evaluation" / "data"
# For simplicity, assuming JSONs are relative to where this script is, or paths are absolute via env.
# If EVAL_PATH is intended to be relative to where the script is, then:
EVAL_PATH = BASE_DIR / "evaluation" / "data" # If evaluation/data is a subdir of where main.py is
# If JSONs are simply in the same dir as main.py:
# EVAL_PATH = BASE_DIR

ONBOARDING_FLOW_DEF_PATH_DEFAULT = EVAL_PATH / "onboarding_flow.json"
ASSESSMENT_FLOW_DEF_PATH_DEFAULT = EVAL_PATH / "assessment_flow.json"
GOLDEN_DATASET_PATH_DEFAULT = EVAL_PATH / "golden_dataset.json"

ONBOARDING_FLOW_DEF_PATH = Path(os.getenv("ONBOARDING_FLOW_DEF_PATH", ONBOARDING_FLOW_DEF_PATH_DEFAULT))
ASSESSMENT_FLOW_DEF_PATH = Path(os.getenv("ASSESSMENT_FLOW_DEF_PATH", ASSESSMENT_FLOW_DEF_PATH_DEFAULT))
GOLDEN_DATASET_PATH = Path(os.getenv("GOLDEN_DATASET_PATH", GOLDEN_DATASET_PATH_DEFAULT))

onboarding_flow_definition_global: Optional[dict] = None
assessment_flow_definition_global: Optional[dict] = None

def load_json_file(file_path: Path) -> Optional[dict | list]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {file_path}: {e}")
    return None

def get_collects_field_from_question_number(q_number: int, flow_def: dict) -> Optional[str]:
    if not flow_def or not isinstance(flow_def.get(tasks.onboarding_flow_id), list):
        logger.warning(f"Invalid onboarding flow definition for flow ID '{tasks.onboarding_flow_id}'")
        return None
    for q_data in flow_def[tasks.onboarding_flow_id]:
        if q_data.get("question_number") == q_number:
            return q_data.get("collects")
    logger.warning(f"Could not find 'collects' for question_number {q_number} in onboarding flow.")
    return None

def _get_simulated_or_interactive_response(
    mode: str, contextualized_question: str, collects_field: Optional[str],
    simulated_responses_map: Optional[dict[str, str]], scenario_id_for_logging: str = "N/A"
) -> Optional[str]:
    if mode == "EVALUATION":
        if not simulated_responses_map:
            logger.warning(f"({scenario_id_for_logging}) Simulated responses map is missing.")
            return None
        if not collects_field or collects_field not in simulated_responses_map:
            logger.warning(
                f"({scenario_id_for_logging}) No simulated response for determined field '{collects_field}'. Q: '{contextualized_question}'"
            )
            return None
        return simulated_responses_map[collects_field]
    elif mode == "INTERACTIVE":
        return input(f"{contextualized_question}\n> ")
    return None

def _core_onboarding_loop(
    user_context: dict[str, Any],
    chat_history: list[str],
    onboarding_flow_def: dict, # Assuming it's loaded
    config: dict[str, Any],
) -> dict[str, Any]:
    mode = config["mode"]
    max_turns = config["max_turns"]
    langfuse_trace = config.get("langfuse_trace") # Optional Langfuse.Trace object
    simulated_responses = config.get("simulated_responses_map") # Optional
    scenario_id = config.get("scenario_id", "N/A_onboarding")

    for turn_num in range(max_turns):
        span = None
        if langfuse_trace and mode == "EVALUATION":
            span = langfuse_trace.span(
                name=f"onboarding-turn-{turn_num + 1}",
                input={"user_context": user_context.copy(), "chat_history": chat_history.copy()},
                metadata={"goal": "Get next onboarding question and process response"}
            )
        
        if mode == "INTERACTIVE": print("-" * 20)
        logger.info(f"({scenario_id}) Onboarding Question Attempt: {turn_num + 1}")

        next_question_data: Optional[dict[str, Any]] = tasks.get_next_onboarding_question(user_context, chat_history)

        if not next_question_data: # Existing logic
            logger.info(f"({scenario_id}) Onboarding flow complete (no next question).")
            if span: span.update(output={"status": "Onboarding ended: no next question"}); span.end()
            break
        
        if not isinstance(next_question_data, dict): # Existing logic
            logger.error(f"({scenario_id}) tasks.get_next_onboarding_question expected dict, got {type(next_question_data)}. Halting.")
            if span: span.update(output={"status": "Error: Task returned non-dict", "output_type": str(type(next_question_data))}); span.end()
            break
            
        contextualized_question = next_question_data.get("contextualized_question")
        # >>> This is where we get the collects_field for the question asked <<<
        collects_field_for_current_q = next_question_data.get("collects_field")
        chosen_q_num = next_question_data.get("chosen_question_number")

        if not collects_field_for_current_q and chosen_q_num is not None:
            collects_field_for_current_q = get_collects_field_from_question_number(chosen_q_num, onboarding_flow_def)
        
        if span: # Existing logic
            span.update(metadata={
                "llm_contextualized_question": contextualized_question,
                "determined_collects_field": collects_field_for_current_q, # Log the field we determined
                "chosen_question_number_from_task": chosen_q_num,
            })

        if not contextualized_question: # Existing logic
            logger.error(f"({scenario_id}) Task data present but no 'contextualized_question'. Halting.")
            if span: span.update(output={"status": "Error: No 'contextualized_question' in task output"}); span.end()
            break

        chat_history.append(f"System to User: {contextualized_question}")
        user_response = _get_simulated_or_interactive_response(
            mode, contextualized_question, collects_field_for_current_q, simulated_responses, scenario_id
        )

        if user_response is None: # Existing logic
            log_msg = f"({scenario_id}) No user response obtained for field '{collects_field_for_current_q}'. "
            if mode == "EVALUATION": log_msg += "Halting onboarding for this scenario."
            else: log_msg += "Halting interactive onboarding."
            logger.warning(log_msg)
            if span: span.update(output={"status": "Halted - no user response", "final_question_asked": contextualized_question}); span.end()
            break
        
        chat_history.append(f"User to System: {user_response}")
        logger.info(f"({scenario_id}) User responded (for {collects_field_for_current_q or 'unknown field'}): {user_response}")

        user_context_before_extraction = user_context.copy()
        # >>> Pass collects_field_for_current_q to extract_onboarding_data_from_response <<<
        user_context = tasks.extract_onboarding_data_from_response(
            user_response,
            user_context,
            chat_history,
            expected_collects_field=collects_field_for_current_q # NEW ARGUMENT
        )
        
        # ... rest of your logging and span update logic for extracted_this_turn ...
        extracted_this_turn: dict[str, Any] = {}
        for k, v in user_context.items():
            if k == 'other' and isinstance(v, dict):
                other_before = user_context_before_extraction.get('other', {})
                other_diff = {k_o: v_o for k_o, v_o in v.items() if other_before.get(k_o) != v_o}
                if other_diff: extracted_this_turn['other'] = other_diff
            elif k not in user_context_before_extraction or user_context_before_extraction[k] != v:
                extracted_this_turn[k] = v
        
        logger.info(f"({scenario_id}) Data extracted/updated: {json.dumps(extracted_this_turn)}")
        if span:
            span.update(output={
                "user_response": user_response, "extracted_data_this_turn": extracted_this_turn,
                "final_user_context_for_turn": user_context.copy()
            })
            span.end()
            
    logger.info(f"({scenario_id}) Onboarding loop finished.")
    return user_context

def _core_assessment_loop(
    user_context: dict[str, Any], chat_history: list[str],
    assessment_flow_def: dict, config: dict[str, Any],
) -> dict[str, Any]:
    mode = config["mode"]
    langfuse_trace = config.get("langfuse_trace")
    simulated_responses = config.get("simulated_responses_map")
    scenario_id = config.get("scenario_id", "N/A_assessment")
    
    questions_to_ask = assessment_flow_def.get(tasks.assessment_flow_id, [])
    if not questions_to_ask:
        logger.error(f"({scenario_id}) Assessment flow definition for ID '{tasks.assessment_flow_id}' is empty/not found.")
        return user_context

    for step_idx, question_def in enumerate(questions_to_ask):
        span = None
        current_q_number = question_def["question_number"]
        collects_field = question_def["collects"]
        raw_q_content = question_def["content"]
        valid_options = question_def["valid_responses"]

        if mode == "EVALUATION":
            if not simulated_responses:
                logger.error(f"({scenario_id}) Simulated responses map missing for eval mode. Halting assessment.")
                break
            if collects_field not in simulated_responses:
                logger.warning(f"({scenario_id}) No simulated response for assessment field '{collects_field}'. Skipping.")
                continue

        if langfuse_trace and mode == "EVALUATION":
            span = langfuse_trace.span(
                name=f"assessment-question-{current_q_number}",
                input={"user_context": user_context.copy(), "current_assessment_step_index": step_idx},
                metadata={"question_number": current_q_number, "collects_field": collects_field, "raw_question_content": raw_q_content}
            )
        
        if mode == "INTERACTIVE": print("-" * 20)
        logger.info(f"({scenario_id}) Assessment Step: Q{current_q_number} (Index {step_idx})")

        task_result: Optional[dict[str, Any]] = tasks.get_assessment_question(
            flow_id=tasks.assessment_flow_id, question_number=current_q_number,
            current_assessment_step=step_idx, user_context=user_context,
            question_to_contextualize=raw_q_content
        )

        if not task_result or 'contextualized_question' not in task_result:
            logger.error(f"({scenario_id}) Failed to get assessment question for step {current_q_number}.")
            if span: span.update(output={"status": "Failed to get contextualized question"}); span.end()
            break 
        
        contextualized_question = task_result["contextualized_question"]
        chat_history.append(f"System to User: {contextualized_question}")
        user_response = _get_simulated_or_interactive_response(
            mode, contextualized_question, collects_field, simulated_responses, scenario_id
        )

        if user_response is None:
            logger.warning(f"({scenario_id}) No user response for assessment Q{current_q_number}. Halting.")
            if span: span.update(output={"status": "Halted - no user response"}); span.end()
            break

        chat_history.append(f"User to System: {user_response}")
        logger.info(f"({scenario_id}) User responded (for {collects_field}): {user_response}")

        validation_result: Optional[dict[str, Any]] = tasks.validate_assessment_answer(
            user_response, current_q_number, valid_responses_options=valid_options
        )
        
        processed_user_response = validation_result.get("processed_user_response") if validation_result else None
        span_output_assessment: dict[str, Any] = {"contextualized_question": contextualized_question, "user_response": user_response}

        if processed_user_response is None:
            logger.warning(f"({scenario_id}) Validation returned no processed response for Q{current_q_number}.")
            span_output_assessment["validation_status"] = "No processed response"
        else:
            logger.info(f"({scenario_id}) Processed response for Q{current_q_number}: {processed_user_response}")
            user_context[collects_field] = processed_user_response
            span_output_assessment["processed_user_response"] = processed_user_response
        
        if span:
            span.update(output=span_output_assessment) # Use update()
            span.update(metadata={"final_user_context_for_turn": user_context.copy()}) # Update metadata
            span.end()
            
    logger.info(f"({scenario_id}) Assessment loop finished.")
    return user_context

def run_interactive_simulation():
    logger.info("--- Starting Interactive Haystack POC Simulation ---")
    user_context: dict[str, Any] = {
        "age": "33", "gender": "female", "goal": "Complete the onboarding process",
        "province": None, "area_type": None, "relationship_status": None,
        "education_level": None, "hunger_days": None, "num_children": None,
        "phone_ownership": None, "other": {}
    }
    chat_history: list[str] = []
    
    if not onboarding_flow_definition_global or not assessment_flow_definition_global:
        logger.error("Flow definitions not loaded globally. Cannot run interactive simulation properly.")
        return

    onboarding_config = {"mode": "INTERACTIVE", "max_turns": 10, "scenario_id": "interactive_onboarding"}
    logger.info("\n--- Simulating Onboarding (Interactive) ---")
    user_context = _core_onboarding_loop(user_context, chat_history, onboarding_flow_definition_global, onboarding_config)

    user_context["goal"] = "Complete the assessment"
    assessment_config = {"mode": "INTERACTIVE", "scenario_id": "interactive_assessment"}
    logger.info("\n--- Simulating Assessment (Interactive) ---")
    user_context = _core_assessment_loop(user_context, chat_history, assessment_flow_definition_global, assessment_config)
    
    logger.info(f"Final interactive context: {json.dumps(user_context, indent=2)}")
    logger.info("--- Interactive Simulation Complete ---")

def run_evaluation_driven_simulation(
    dataset_item: dict[str, Any], langfuse_instance: Langfuse,
    onboarding_flow_def: dict, assessment_flow_def: dict,
) -> Optional[str]:
    scenario_id = dataset_item.get("scenario_id", "unknown_scenario")
    trace = langfuse_instance.trace(
        name=f"eval-simulation-{scenario_id}",
        user_id=dataset_item.get("user_persona", {}).get("id", "simulated_user"),
        metadata={"description": dataset_item.get("description", ""), "scenario_data": dataset_item},
        tags=[dataset_item.get("flow_type", "unknown_flow"), "evaluation_run"]
    )

    base_user_context: dict[str, Any] = {
        "age": None, "gender": None, "province": None, "area_type": None,
        "relationship_status": None, "education_level": None,
        "hunger_days": None, "num_children": None, "phone_ownership": None, "other": {}
    }
    user_context = {**base_user_context, **dataset_item.get("user_persona", {}).get("persona_details_from_brief", {})}
    user_context.update(dataset_item.get("initial_user_context", {}))
    chat_history: list[str] = []
    if dataset_item.get("initial_user_utterance"):
        chat_history.append(f"User to System: {dataset_item['initial_user_utterance']}")

    simulated_responses = dataset_item.get("simulated_user_responses_map", {})
    eval_config_base = {
        "mode": "EVALUATION", "langfuse_trace": trace,
        "simulated_responses_map": simulated_responses, "scenario_id": scenario_id
    }

    if dataset_item["flow_type"] == tasks.onboarding_flow_id:
        logger.info(f"--- Simulating Onboarding (Evaluation) for {scenario_id} ---")
        user_context["goal"] = "Complete the onboarding process"
        onboarding_config = {**eval_config_base, "max_turns": len(simulated_responses) + 3} # a bit more leeway
        user_context = _core_onboarding_loop(user_context, chat_history, onboarding_flow_def, onboarding_config)
    elif dataset_item["flow_type"] == tasks.assessment_flow_id:
        logger.info(f"--- Simulating Assessment (Evaluation) for {scenario_id} ---")
        user_context["goal"] = "Complete the assessment"
        assessment_config = {**eval_config_base}
        user_context = _core_assessment_loop(user_context, chat_history, assessment_flow_def, assessment_config)
    else:
        logger.warning(f"Unknown flow_type: {dataset_item['flow_type']} for scenario {scenario_id}")
        trace.update(metadata={"error": f"Unknown flow_type: {dataset_item['flow_type']}"}) # Use update

    logger.info(f"--- EVALUATION-DRIVEN Simulation Complete for: {scenario_id} ---")
    logger.info(f"Final User Context for {scenario_id}: {json.dumps(user_context, indent=2)}")
    trace.update(output={"final_user_context": user_context, "final_chat_history": chat_history}) # Use update
    return trace.id

def _log_critical_error_to_langfuse(
    langfuse_instance: Optional[Langfuse], scenario_id: str,
    dataset_item_flow_type: Optional[str], e: Exception
):
    if not langfuse_instance: return
    try:
        error_payload = {"error_message": str(e), "full_traceback": traceback.format_exc()}
        langfuse_instance.trace(
            name=f"critical-error-simulation-{scenario_id}", user_id="system_error",
            metadata={"dataset_item_id": scenario_id, **error_payload}, # Error details in metadata
            tags=["critical_error", dataset_item_flow_type if dataset_item_flow_type else "unknown_flow"]
        )
    except Exception as log_e:
        logger.error(f"Failed to log critical error to Langfuse itself: {log_e}", exc_info=True)

def main():
    run_mode = os.getenv("RUN_MODE", "EVALUATION").upper()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    global onboarding_flow_definition_global, assessment_flow_definition_global
    onboarding_flow_definition_global = load_json_file(ONBOARDING_FLOW_DEF_PATH)
    assessment_flow_definition_global = load_json_file(ASSESSMENT_FLOW_DEF_PATH)

    if not onboarding_flow_definition_global or not assessment_flow_definition_global:
        logger.error("Critical: Failed to load flow definitions. Cannot proceed.")
        return

    langfuse_instance: Optional[Langfuse] = None
    if run_mode == "EVALUATION":
        logger.info("RUN_MODE=EVALUATION: Initializing Langfuse and loading dataset...")
        try:
            langfuse_instance = Langfuse()
            if not langfuse_instance.auth_check():
                 logger.error("Langfuse authentication failed. Check credentials/host.")
                 return
            logger.info("Langfuse client initialized for EVALUATION mode.")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse for EVALUATION: {e}.", exc_info=True)
            return

        golden_dataset = load_json_file(GOLDEN_DATASET_PATH)
        if not golden_dataset or not isinstance(golden_dataset, list):
            logger.error(f"Failed to load golden dataset from {GOLDEN_DATASET_PATH}. Exiting.")
            return
        logger.info(f"Loaded {len(golden_dataset)} scenarios for EVALUATION.")

        for i, dataset_item in enumerate(golden_dataset):
            scenario_id = dataset_item.get("scenario_id", f"unknown_eval_scenario_{i+1}")
            logger.info(f"Processing EVALUATION scenario {i+1}/{len(golden_dataset)}: {scenario_id}")
            try:
                if langfuse_instance:
                    trace_id = run_evaluation_driven_simulation(
                        dataset_item, langfuse_instance,
                        onboarding_flow_definition_global, assessment_flow_definition_global
                    )
                    if trace_id:
                        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                        logger.info(f"EVALUATION of {scenario_id} complete. Trace: {host}/trace/{trace_id}")
            except Exception as e:
                logger.error(f"Critical error in EVALUATION of {scenario_id}: {e}", exc_info=True)
                _log_critical_error_to_langfuse(langfuse_instance, scenario_id, dataset_item.get("flow_type"), e)
        logger.info("All EVALUATION scenarios processed.")
        if langfuse_instance: langfuse_instance.flush()

    elif run_mode == "INTERACTIVE":
        logger.info("RUN_MODE=INTERACTIVE: Starting interactive simulation...")
        run_interactive_simulation()
    else:
        logger.error(f"Unknown RUN_MODE: '{run_mode}'. Set to 'EVALUATION' or 'INTERACTIVE'.")

    logger.info("Main processing finished. Attempting resource cleanup...")
    doc_store.close_weaviate_client_connection()
    logger.info("Script execution complete.")


if __name__ == "__main__":
    main()