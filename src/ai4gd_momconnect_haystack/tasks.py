# tasks.py
import json
import logging
from typing import Any, Dict, List, Optional # Python 3.9+ dict, list

from langfuse.decorators import observe # The decorator

from . import doc_store
from . import pipelines

logger = logging.getLogger(__name__)

# --- Configuration ---
onboarding_flow_id: str = "onboarding"
assessment_flow_id: str = "dma-assessment"

all_onboarding_questions: list[dict[str, Any]] = doc_store.ONBOARDING_FLOWS.get(onboarding_flow_id, [])
all_assessment_questions: list[dict[str, Any]] = doc_store.ASSESSMENT_FLOWS.get(assessment_flow_id, [])


@observe()
def get_next_onboarding_question(user_context: dict[str, Any], chat_history: list[str]) -> Optional[dict[str, Any]]:
    """
    Determines and contextualizes the next onboarding question.
    Relies on the pipeline to return a dictionary with chosen_question_number,
    contextualized_question, collects_field, and fallback_used.
    """
    remaining_questions_list = doc_store.get_remaining_onboarding_questions(user_context, all_onboarding_questions)

    if not remaining_questions_list:
        logger.info("All onboarding questions answered. Onboarding complete.")
        return None

    logger.info("Running next question selection pipeline...")
    onboarding_question_pipeline = pipelines.create_next_onboarding_question_pipeline()
    
    next_question_result_dict = pipelines.run_next_onboarding_question_pipeline(
        onboarding_question_pipeline, user_context, remaining_questions_list, chat_history
    )

    if not next_question_result_dict or not next_question_result_dict.get("contextualized_question"):
        logger.error("Pipeline failed to return a valid next question structure. Onboarding cannot proceed with this turn.")
        return None

    # Log for local debugging; Langfuse captures the full return dict as this function's output
    chosen_q_num = next_question_result_dict.get("chosen_question_number")
    actual_q_def = next((q for q in all_onboarding_questions if q['question_number'] == chosen_q_num), None)
    original_content = actual_q_def['content'] if actual_q_def else "Unknown original question"

    logger.info(f"Pipeline chose question number: {chosen_q_num} (Original: '{original_content}')")
    logger.info(f"Pipeline contextualized question: {next_question_result_dict.get('contextualized_question')}")
    if next_question_result_dict.get("fallback_used"):
        logger.warning("Pipeline reported using fallback logic for next question.")

    return next_question_result_dict


@observe()
def extract_onboarding_data_from_response(
    user_response: str,
    user_context: dict[str, Any],
    chat_history: list[str],
    expected_collects_field: Optional[str] = None # NEW PARAMETER
) -> dict[str, Any]:
    """
    Extracts structured data from a user's response during onboarding
    and updates the user_context.
    'expected_collects_field' helps in guiding mock data if mocks are active.
    """
    logger.info(f"Running data extraction pipeline (expecting data for: {expected_collects_field or 'any field'})...")
    onboarding_data_extraction_pipe = pipelines.create_onboarding_data_extraction_pipeline()
    
    extracted_data_from_pipeline = pipelines.run_onboarding_data_extraction_pipeline(
        onboarding_data_extraction_pipe,
        user_response,
        user_context,
        chat_history,
        expected_collects_field=expected_collects_field # PASS DOWN
    )

    updated_user_context = user_context.copy()

    if extracted_data_from_pipeline and isinstance(extracted_data_from_pipeline, dict):
        logger.info(f"[Raw Extracted Data for user_response='{user_response}']: "
                    f"{json.dumps(extracted_data_from_pipeline, indent=2)}")
        
        onboarding_data_to_collect = [
            "province", "area_type", "relationship_status", "education_level",
            "hunger_days", "num_children", "phone_ownership"
        ]
        actually_updated_fields = {}
        for key, value in extracted_data_from_pipeline.items():
            if key in onboarding_data_to_collect:
                if updated_user_context.get(key) != value:
                    updated_user_context[key] = value
                    actually_updated_fields[key] = value
            else: 
                if "other" not in updated_user_context: updated_user_context["other"] = {}
                if updated_user_context["other"].get(key) != value:
                    updated_user_context["other"][key] = value
                    if "other" not in actually_updated_fields: actually_updated_fields["other"] = {}
                    actually_updated_fields["other"][key] = value
        
        if actually_updated_fields:
             logger.info(f"Fields updated in user_context: {json.dumps(actually_updated_fields)}")
    else:
        logger.warning(f"Data extraction pipeline did not produce a valid result for user_response='{user_response}'.")
        
    return updated_user_context


@observe()
def get_assessment_question(
    flow_id: str,
    question_number: int, # The question_number from the flow definition
    current_assessment_step: int, # 0-based index
    user_context: dict[str, Any],
    question_to_contextualize: str # Raw question content, used as fallback
) -> Optional[dict[str, Any]]:
    if flow_id != assessment_flow_id:
        logger.warning(f"get_assessment_question called with flow_id='{flow_id}' but using internal '{assessment_flow_id}'.")
        # Use the internal assessment_flow_id for safety

    if current_assessment_step >= len(all_assessment_questions):
        logger.info("Assessment flow complete (all questions processed).")
        return None

    next_step_data = all_assessment_questions[current_assessment_step]
    actual_question_number_from_flow = next_step_data['question_number']

    if actual_question_number_from_flow != question_number:
        logger.warning(
            f"Mismatch: Expected Q#{question_number} but current step ({current_assessment_step}) "
            f"is Q#{actual_question_number_from_flow}."
        )
    logger.info(f"Processing assessment Q#{actual_question_number_from_flow}. Original: '{question_to_contextualize}'")

    assessment_context_pipeline = pipelines.create_assessment_contextualization_pipeline()
    contextualized_q_str = pipelines.run_assessment_contextualization_pipeline(
        assessment_context_pipeline,
        assessment_flow_id, # Pass the correct flow_id
        actual_question_number_from_flow, # Use number from current step data
        user_context
    )

    final_contextualized_question = contextualized_q_str
    if not contextualized_q_str:
        logger.warning(f"Contextualization failed for Q{actual_question_number_from_flow}. Using raw content.")
        final_contextualized_question = question_to_contextualize
        
    return {
        "contextualized_question": final_contextualized_question,
        "current_question_number": actual_question_number_from_flow
    }


@observe()
def validate_assessment_answer(
    user_response: str,
    current_question_number: int,
    valid_responses_options: Optional[list[str]] = None
) -> Optional[dict[str, Any]]:
    logger.info(f"Running assessment response validation pipeline for Q{current_question_number}...")
    validator_pipe = pipelines.create_assessment_response_validator_pipeline()
    processed_response_str = pipelines.run_assessment_response_validator_pipeline(
        validator_pipe, user_response # Pass other args if pipeline uses them
    )

    final_processed_response = None
    if processed_response_str is None:
        logger.warning(f"Validation pipeline returned None for Q{current_question_number}.")
    elif processed_response_str.lower() == "nonsense":
        logger.info(f"Response to Q{current_question_number} ('{user_response}') validated as 'nonsense'.")
        # final_processed_response remains None
    else:
        logger.info(f"Validated response for Q{current_question_number}: '{processed_response_str}' from original '{user_response}'")
        final_processed_response = processed_response_str

    return {
        "processed_user_response": final_processed_response,
        "current_assessment_step": current_question_number
    }