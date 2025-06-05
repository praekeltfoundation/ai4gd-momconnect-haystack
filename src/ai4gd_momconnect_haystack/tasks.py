import logging
import json
from typing import Any

from . import doc_store
from . import pipelines

# --- Configuration ---
logger = logging.getLogger(__name__)

onboarding_flow_id = "onboarding"
assessment_flow_id = "dma-assessment"

all_onboarding_questions = doc_store.ONBOARDING_FLOWS.get(onboarding_flow_id, [])
all_assessment_questions = doc_store.ASSESSMENT_FLOWS.get(assessment_flow_id, [])


def get_next_onboarding_question(user_context: dict, chat_history: list) -> str | None:
    # Get remaining questions
    remaining_questions_list = doc_store.get_remaining_onboarding_questions(
        user_context, all_onboarding_questions
    )

    if not remaining_questions_list:
        logger.info("All onboarding questions answered. Onboarding complete.")
        return None

    # LLM decides the next question
    logger.info("Running next question selection pipeline...")
    next_question_pipeline = pipelines.create_next_onboarding_question_pipeline()
    next_question_result = pipelines.run_next_onboarding_question_pipeline(
        next_question_pipeline, user_context, remaining_questions_list, chat_history
    )
    if not next_question_result:
        logger.error(
            "Failed to get next question from the pipeline. Ending onboarding."
        )
        return None
    chosen_question_number = next_question_result.get("chosen_question_number")
    contextualized_question = next_question_result.get("contextualized_question")
    if chosen_question_number is None or contextualized_question is None:
        logger.error("Could not determine next question. Ending onboarding.")
        return None

    # Find the actual question data using the chosen_question_number, simply for comparison during simulation/development.
    current_step_data = next(
        (
            q
            for q in all_onboarding_questions
            if q["question_number"] == chosen_question_number
        ),
        None,
    )

    if not current_step_data:
        logger.error(
            f"Could not find question data for question_number {chosen_question_number}. Skipping."
        )
    else:
        logger.info(f"Original question was: '{current_step_data['content']}'")

    return contextualized_question


def extract_onboarding_data_from_response(
    user_response: str, user_context: dict, chat_history: list
) -> dict:
    logger.info("Running data extraction pipeline...")
    onboarding_data_extraction_pipe = (
        pipelines.create_onboarding_data_extraction_pipeline()
    )
    extracted_data = pipelines.run_onboarding_data_extraction_pipeline(
        onboarding_data_extraction_pipe, user_response, user_context, chat_history
    )

    print(f"[Extracted Data]:\n{json.dumps(extracted_data, indent=2)}\n")

    if extracted_data:
        # Store extracted data:
        onboarding_data_to_collect = [
            "province",
            "area_type",
            "relationship_status",
            "education_level",
            "hunger_days",
            "num_children",
            "phone_ownership",
        ]
        for k, v in extracted_data.items():
            logger.info(f"Extracted {k}: {v}")
            if k in onboarding_data_to_collect:
                user_context[k] = v
                logger.info(f"Updated user_context for {k}: {v}")
            else:
                user_context["other"][k] = v
    else:
        logger.warning("Data extraction pipeline did not produce a result.")
    return user_context


def get_assessment_question(
    flow_id: str, question_number: int, current_assessment_step: int, user_context: dict
) -> dict:
    if current_assessment_step >= len(all_assessment_questions):
        logger.info("Assessment flow complete.")
        return {}

    next_step_data = all_assessment_questions[current_assessment_step]
    current_question_number = next_step_data["question_number"]
    logger.info(
        f"Processing step {current_question_number} for flow '{assessment_flow_id}'"
    )

    # Contextualize the current question
    logger.info("Running contextualization pipeline...")
    # Call with correct arguments: pipeline, flow_id, question_number, user_context
    assessment_contextualization_pipe = (
        pipelines.create_assessment_contextualization_pipeline()
    )
    contextualized_question = pipelines.run_assessment_contextualization_pipeline(
        assessment_contextualization_pipe,
        assessment_flow_id,
        current_question_number,
        user_context,
    )

    if not contextualized_question:
        # Fallback
        logger.warning(
            f"Contextualization failed for step {current_question_number}. Using raw content."
        )
        contextualized_question = next_step_data["content"]
        print(f"\n[System to User (Fallback)]:\n{contextualized_question}\n")
    return {
        "contextualized_question": contextualized_question,
        "current_question_number": current_question_number,
    }


def validate_assessment_answer(
    user_response: str, current_question_number: int
) -> dict[str, Any]:
    validator_pipe = pipelines.create_assessment_response_validator_pipeline()
    processed_user_response = pipelines.run_assessment_response_validator_pipeline(
        validator_pipe, user_response
    )

    if not processed_user_response:
        logger.warning(
            f"Response validation failed for step {current_question_number}."
        )

    # Move to the next step, or try again if the response was invalid
    if processed_user_response:
        logger.info(
            f"Storing validated response for question_number {current_question_number}: {processed_user_response}"
        )
        current_assessment_step = current_question_number
    return {
        "processed_user_response": processed_user_response,
        "current_assessment_step": current_assessment_step,
    }
