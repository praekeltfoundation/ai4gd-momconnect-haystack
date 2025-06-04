import json
import logging
import os
import sys
import traceback  # For logging critical errors to Langfuse
from typing import Any, Optional, Union
from pathlib import Path

from langfuse import Langfuse # Conditionally used

# Assuming tasks.py is in the same directory or discoverable package
from . import tasks

logger = logging.getLogger(__name__)

# --- Configuration and File Loading ---
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "evaluation" / "data"

def load_json_file(file_path: str) -> Optional[dict | list]: # Using dict | list for Python 3.10+
    """Helper to load a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
    return None

ONBOARDING_FLOW_DEF_PATH = os.getenv("ONBOARDING_FLOW_DEF_PATH", EVAL_PATH / "onboarding_flow.json")
ASSESSMENT_FLOW_DEF_PATH = os.getenv("ASSESSMENT_FLOW_DEF_PATH", EVAL_PATH / "assessment_flow.json")
GOLDEN_DATASET_PATH = os.getenv("GOLDEN_DATASET_PATH", EVAL_PATH / "golden_dataset.json")

onboarding_flow_definition_global: Optional[dict] = None
assessment_flow_definition_global: Optional[dict] = None

def get_collects_field_from_question_number(q_number: int, flow_def: dict) -> Optional[str]:
    """Finds 'collects' field using question_number in an onboarding flow definition."""
    if not flow_def or not isinstance(flow_def.get(tasks.onboarding_flow_id), list):
        logger.warning(f"Invalid onboarding flow definition for flow ID '{tasks.onboarding_flow_id}'")
        return None
    for q_data in flow_def[tasks.onboarding_flow_id]:
        if q_data.get("question_number") == q_number:
            return q_data.get("collects")
    logger.warning(f"Could not find 'collects' for question_number {q_number} in onboarding flow.")
    return None

# --- Core Simulation Logic ---

def _get_simulated_or_interactive_response(
    mode: str,
    contextualized_question: str,
    collects_field: Optional[str],
    simulated_responses_map: Optional[dict[str, str]],
    scenario_id_for_logging: str = "N/A"
) -> Optional[str]:
    """Gets user response either from simulated map or interactive input."""
    if mode == "EVALUATION":
        if not simulated_responses_map: # Guard against None
            logger.warning(f"({scenario_id_for_logging}) Simulated responses map is missing.")
            return None
        if not collects_field or collects_field not in simulated_responses_map:
            logger.warning(
                f"({scenario_id_for_logging}) No simulated response for determined field '{collects_field}' "
                f"or field not determined. Question: '{contextualized_question}'"
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
    """Internal core logic for the onboarding simulation loop."""
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

        # CRITICAL ASSUMPTION: tasks.get_next_onboarding_question returns a dict as specified
        next_question_data: Optional[dict[str, Any]] = tasks.get_next_onboarding_question(user_context, chat_history)

        if not next_question_data:
            logger.info(f"({scenario_id}) Onboarding flow complete (no next question).")
            if span: span.output({"status": "Onboarding ended: no next question"}); span.end()
            break
        
        if not isinstance(next_question_data, dict):
            logger.error(f"({scenario_id}) tasks.get_next_onboarding_question returned non-dict: {type(next_question_data)}. Halting.")
            if span: span.output({"status": "Error: Task returned non-dict", "output_type": str(type(next_question_data))}); span.end()
            break
            
        contextualized_question = next_question_data.get("contextualized_question")
        collects_field = next_question_data.get("collects_field")
        chosen_q_num = next_question_data.get("chosen_question_number")

        if not collects_field and chosen_q_num is not None:
            collects_field = get_collects_field_from_question_number(chosen_q_num, onboarding_flow_def)
        
        if span and isinstance(span.metadata, dict): # Ensure metadata is a dict before **
            span.metadata.update({
                "llm_contextualized_question": contextualized_question,
                "determined_collects_field": collects_field,
                "chosen_question_number_from_task": chosen_q_num,
            })

        if not contextualized_question:
            logger.error(f"({scenario_id}) Task data present but no 'contextualized_question'. Halting.")
            if span: span.output({"status": "Error: No 'contextualized_question' in task output"}); span.end()
            break

        chat_history.append(f"System to User: {contextualized_question}")
        user_response = _get_simulated_or_interactive_response(
            mode, contextualized_question, collects_field, simulated_responses, scenario_id
        )

        if user_response is None:
            log_msg = f"({scenario_id}) No user response obtained for field '{collects_field}'. "
            if mode == "EVALUATION": log_msg += "Halting onboarding for this scenario."
            else: log_msg += "Halting interactive onboarding."
            logger.warning(log_msg)
            if span: span.output({"status": "Halted - no user response", "final_question_asked": contextualized_question}); span.end()
            break
        
        chat_history.append(f"User to System: {user_response}")
        logger.info(f"({scenario_id}) User responded (for {collects_field or 'unknown field'}): {user_response}")

        user_context_before_extraction = user_context.copy()
        user_context = tasks.extract_onboarding_data_from_response(user_response, user_context, chat_history)
        
        extracted_this_turn: dict[str, Any] = {
            k: v for k, v in user_context.items()
            if k not in user_context_before_extraction or user_context_before_extraction[k] != v
        }
        if 'other' in user_context:
            other_before = user_context_before_extraction.get('other', {})
            other_after = user_context['other']
            other_diff = {k_o: v_o for k_o, v_o in other_after.items() if other_before.get(k_o) != v_o}
            if other_diff: extracted_this_turn['other'] = other_diff
        
        logger.info(f"({scenario_id}) Data extracted/updated: {json.dumps(extracted_this_turn)}")
        if span:
            span.output({
                "user_response": user_response, "extracted_data_this_turn": extracted_this_turn,
                "final_user_context_for_turn": user_context.copy()
            })
            span.end()
            
    logger.info(f"({scenario_id}) Onboarding loop finished.")
    return user_context

def _core_assessment_loop(
    user_context: dict[str, Any],
    chat_history: list[str],
    assessment_flow_def: dict, # Assuming it's loaded
    config: dict[str, Any],
) -> dict[str, Any]:
    """Internal core logic for the assessment simulation loop."""
    mode = config["mode"]
    langfuse_trace = config.get("langfuse_trace") # Optional
    simulated_responses = config.get("simulated_responses_map") # Optional
    scenario_id = config.get("scenario_id", "N/A_assessment")
    
    questions_to_ask = assessment_flow_def.get(tasks.assessment_flow_id, [])
    if not questions_to_ask:
        logger.error(f"({scenario_id}) Assessment flow definition empty or not found for ID '{tasks.assessment_flow_id}'. Cannot proceed.")
        return user_context

    for step_idx, question_def in enumerate(questions_to_ask):
        span = None
        current_q_number = question_def["question_number"]
        collects_field = question_def["collects"]
        raw_q_content = question_def["content"]
        valid_options = question_def["valid_responses"]

        if mode == "EVALUATION":
            if not simulated_responses: # Guard
                logger.error(f"({scenario_id}) Simulated responses map missing for evaluation mode. Halting assessment.")
                break
            if collects_field not in simulated_responses:
                logger.warning(f"({scenario_id}) No simulated response for assessment field '{collects_field}'. Skipping.")
                continue

        if langfuse_trace and mode == "EVALUATION":
            span = langfuse_trace.span(
                name=f"assessment-question-{current_q_number}",
                input={"user_context": user_context.copy(), "current_assessment_step_index": step_idx},
                metadata={
                    "question_number": current_q_number, "collects_field": collects_field,
                    "raw_question_content": raw_q_content
                }
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
            if span: span.output({"status": "Failed to get contextualized question"}); span.end()
            break 
        
        contextualized_question = task_result["contextualized_question"]
        chat_history.append(f"System to User: {contextualized_question}")
        user_response = _get_simulated_or_interactive_response(
            mode, contextualized_question, collects_field, simulated_responses, scenario_id
        )

        if user_response is None: # Could happen if interactive input fails or eval mapping fails
            logger.warning(f"({scenario_id}) No user response obtained for assessment Q{current_q_number}. Halting assessment.")
            if span: span.output({"status": "Halted - no user response"}); span.end()
            break

        chat_history.append(f"User to System: {user_response}")
        logger.info(f"({scenario_id}) User responded (for {collects_field}): {user_response}")

        validation_result: Optional[dict[str, Any]] = tasks.validate_assessment_answer(
            user_response, current_q_number, valid_responses_options=valid_options
        )
        
        processed_user_response = validation_result.get("processed_user_response") if validation_result else None

        span_output_assessment: dict[str, Any] = {"contextualized_question": contextualized_question, "user_response": user_response}
        if processed_user_response is None:
            logger.warning(f"({scenario_id}) Response validation returned no processed response for Q{current_q_number}.")
            span_output_assessment["validation_status"] = "No processed response"
        else:
            logger.info(f"({scenario_id}) Processed response for Q{current_q_number}: {processed_user_response}")
            user_context[collects_field] = processed_user_response # Update context
            span_output_assessment["processed_user_response"] = processed_user_response
        
        if span:
            span.output(span_output_assessment)
            if isinstance(span.metadata, dict): # Ensure metadata is a dict
                span.metadata.update({"final_user_context_for_turn": user_context.copy()})
            span.end()
            
    logger.info(f"({scenario_id}) Assessment loop finished.")
    return user_context

# --- Main Simulation Functions (Wrappers) ---

def run_interactive_simulation():
    """Runs an interactive simulation."""
    logger.info("--- Starting Interactive Haystack POC Simulation ---")
    # Initial context for interactive mode
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

    onboarding_config = {
        "mode": "INTERACTIVE",
        "max_turns": 10, # Default for interactive onboarding
        "scenario_id": "interactive_onboarding"
    }
    logger.info("\n--- Simulating Onboarding (Interactive) ---")
    user_context = _core_onboarding_loop(user_context, chat_history, onboarding_flow_definition_global, onboarding_config)

    user_context["goal"] = "Complete the assessment" # Switch goal
    assessment_config = {
        "mode": "INTERACTIVE",
        "scenario_id": "interactive_assessment"
        # max_turns for assessment is implicitly the number of questions in the flow_def
    }
    logger.info("\n--- Simulating Assessment (Interactive) ---")
    user_context = _core_assessment_loop(user_context, chat_history, assessment_flow_definition_global, assessment_config)
    
    logger.info(f"Final interactive context: {json.dumps(user_context, indent=2)}")
    logger.info("--- Interactive Simulation Complete ---")


def run_evaluation_driven_simulation(
    dataset_item: dict[str, Any],
    langfuse_instance: Langfuse, # Langfuse.Trace object
    onboarding_flow_def: dict,
    assessment_flow_def: dict,
) -> Optional[str]: # Returns trace_id
    """Runs a data-driven simulation for evaluation with Langfuse tracing."""
    scenario_id = dataset_item.get("scenario_id", "unknown_scenario")
    # The trace object is created here and passed into core loops via config
    trace = langfuse_instance.trace(
        name=f"eval-simulation-{scenario_id}",
        user_id=dataset_item.get("user_persona", {}).get("id", "simulated_user"),
        metadata={"description": dataset_item.get("description", ""), "scenario_data": dataset_item},
        tags=[dataset_item.get("flow_type", "unknown_flow"), "evaluation_run"]
    )

    base_user_context: dict[str, Any] = { # Define base structure
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
    
    # Common config for both evaluation flows
    eval_config_base = {
        "mode": "EVALUATION",
        "langfuse_trace": trace,
        "simulated_responses_map": simulated_responses,
        "scenario_id": scenario_id
    }

    if dataset_item["flow_type"] == tasks.onboarding_flow_id:
        logger.info(f"--- Simulating Onboarding (Evaluation) for {scenario_id} ---")
        user_context["goal"] = "Complete the onboarding process"
        onboarding_config = {**eval_config_base, "max_turns": len(simulated_responses) + 2}
        user_context = _core_onboarding_loop(user_context, chat_history, onboarding_flow_def, onboarding_config)
    elif dataset_item["flow_type"] == tasks.assessment_flow_id:
        logger.info(f"--- Simulating Assessment (Evaluation) for {scenario_id} ---")
        user_context["goal"] = "Complete the assessment"
        # Assessment config doesn't strictly need max_turns if driven by flow_def length
        assessment_config = {**eval_config_base} 
        user_context = _core_assessment_loop(user_context, chat_history, assessment_flow_def, assessment_config)
    else:
        logger.warning(f"Unknown flow_type: {dataset_item['flow_type']} for scenario {scenario_id}")
        if isinstance(trace.metadata, dict): # Ensure metadata is a dict
            trace.metadata.update({"error": f"Unknown flow_type: {dataset_item['flow_type']}"})

    logger.info(f"--- EVALUATION-DRIVEN Simulation Complete for: {scenario_id} ---")
    logger.info(f"Final User Context for {scenario_id}: {json.dumps(user_context, indent=2)}")
    trace.output({"final_user_context": user_context, "final_chat_history": chat_history})
    return trace.id

# --- Main Orchestration ---

def _log_critical_error_to_langfuse(
    langfuse_instance: Optional[Langfuse], # Can be None if not in eval mode
    scenario_id: str,
    dataset_item_flow_type: str,
    e: Exception
):
    """Helper to log critical errors to Langfuse if instance is available."""
    if not langfuse_instance: return
    try:
        error_trace = langfuse_instance.trace(
            name=f"critical-error-simulation-{scenario_id}", user_id="system_error",
            metadata={"error_message": str(e), "dataset_item_id": scenario_id},
            tags=["critical_error", dataset_item_flow_type]
        )
        error_trace.output({"error_details": str(e), "full_traceback": traceback.format_exc()})
    except Exception as log_e:
        logger.error(f"Failed to log critical error to Langfuse: {log_e}")

def main():
    """Main function to drive simulations based on RUN_MODE."""
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

    langfuse_instance: Optional[Langfuse] = None # Initialize to None
    if run_mode == "EVALUATION":
        logger.info("RUN_MODE=EVALUATION: Initializing Langfuse and loading dataset...")
        try:
            langfuse_instance = Langfuse() # LANGFUSE_PUBLIC_KEY, SECRET_KEY, HOST from env
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
                # langfuse_instance is guaranteed to be non-None here if we reached this point
                trace_id = run_evaluation_driven_simulation(
                    dataset_item, langfuse_instance, # type: ignore
                    onboarding_flow_definition_global, assessment_flow_definition_global
                )
                if trace_id:
                    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                    logger.info(f"EVALUATION of {scenario_id} complete. Trace: {host}/trace/{trace_id}")
            except Exception as e:
                logger.error(f"Critical error in EVALUATION of {scenario_id}: {e}", exc_info=True)
                _log_critical_error_to_langfuse(langfuse_instance, scenario_id, dataset_item.get("flow_type", "unknown"), e)
        logger.info("All EVALUATION scenarios processed.")
        if langfuse_instance: langfuse_instance.flush()

    elif run_mode == "INTERACTIVE":
        logger.info("RUN_MODE=INTERACTIVE: Starting interactive simulation...")
        run_interactive_simulation() # Langfuse not passed, @observe in tasks will be benign if client not active
    else:
        logger.error(f"Unknown RUN_MODE: '{run_mode}'. Set to 'EVALUATION' or 'INTERACTIVE'.")

if __name__ == "__main__":
    main()