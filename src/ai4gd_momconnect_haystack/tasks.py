import json
import logging

from langfuse.decorators import observe

# Assuming . is the current package/directory
from . import doc_store, pipelines

# --- Configuration ---
logger = logging.getLogger(__name__)

ONBOARDING_FLOW_ID = "onboarding"
ASSESSMENT_FLOW_ID = "dma-assessment"

# These are loaded at module level
all_onboarding_questions = doc_store.ONBOARDING_FLOWS.get(
    ONBOARDING_FLOW_ID, []
)
all_assessment_questions = doc_store.ASSESSMENT_FLOWS.get(
    ASSESSMENT_FLOW_ID, []
)


@observe()
def get_next_onboarding_question(
    user_context: dict, chat_history: list
) -> str | None:
    """
    Determines and contextualizes the next onboarding question.
    """
    remaining_questions_list = (
        doc_store.get_remaining_onboarding_questions(
            user_context, all_onboarding_questions
        )
    )

    try:
        current_observation = observe.get_current_observation()
        if current_observation:
            current_observation.metadata(
                {
                    "remaining_questions_count": len(remaining_questions_list),
                    "remaining_question_numbers": [
                        q["question_number"] for q in remaining_questions_list
                    ],
                }
            )
    except Exception as e:
        logger.warning(f"Could not get current Langfuse observation: {e}")

    if not remaining_questions_list:
        logger.info("All onboarding questions answered. Onboarding complete.")
        return None

    logger.info("Running next question selection pipeline...")
    next_question_pipeline = (
        pipelines.create_next_onboarding_question_pipeline()
    )
    next_question_result = pipelines.run_next_onboarding_question_pipeline(
        next_question_pipeline,
        user_context,
        remaining_questions_list,
        chat_history,
    )

    if not next_question_result:
        logger.error(
            "Failed to get next question from the pipeline. Ending onboarding."
        )
        return None

    chosen_question_number = next_question_result.get("chosen_question_number")
    contextualized_question = next_question_result.get(
        "contextualized_question"
    )

    if chosen_question_number is None or contextualized_question is None:
        logger.error("Could not determine next question. Ending onboarding.")
        return None

    current_step_data = next(
        (
            q
            for q in all_onboarding_questions
            if q["question_number"] == chosen_question_number
        ),
        None,
    )
    original_question_content = None
    if not current_step_data:
        logger.error(
            f"Could not find question data for question_number"
            f" {chosen_question_number}. Skipping."
        )
    else:
        original_question_content = current_step_data["content"]
        logger.info(f"Original question was: '{original_question_content}'")

    try:
        current_observation = observe.get_current_observation()
        if current_observation:
            # Preserve existing metadata if any was set earlier in the observation
            existing_metadata = current_observation.metadata or {}
            current_observation.metadata(
                {
                    **existing_metadata,
                    "chosen_question_number_from_pipeline": (
                        chosen_question_number
                    ),
                    "original_question_content": original_question_content,
                }
            )
    except Exception as e:
        logger.warning(f"Could not update Langfuse observation metadata: {e}")

    return contextualized_question


@observe()
def extract_onboarding_data_from_response(
    user_response: str, user_context: dict, chat_history: list
) -> dict:
    """
    Extracts structured data from a user's response during onboarding.
    """
    logger.info("Running data extraction pipeline...")
    onboarding_data_extraction_pipe = (
        pipelines.create_onboarding_data_extraction_pipeline()
    )
    extracted_data_from_pipeline = (
        pipelines.run_onboarding_data_extraction_pipeline(
            onboarding_data_extraction_pipe,
            user_response,
            user_context,
            chat_history,
        )
    )

    try:
        current_observation = observe.get_current_observation()
        if current_observation:
            current_observation.metadata(
                {
                    "raw_extracted_data_from_pipeline": (
                        extracted_data_from_pipeline
                    ),
                }
            )
    except Exception as e:
        logger.warning(f"Could not update Langfuse observation metadata: {e}")

    updated_user_context = user_context.copy()

    if extracted_data_from_pipeline:
        logger.info(
            f"[Raw Extracted Data from LLM]:\n"
            f"{json.dumps(extracted_data_from_pipeline, indent=2)}\n"
        )
        onboarding_data_to_collect = [
            "province",
            "area_type",
            "relationship_status",
            "education_level",
            "hunger_days",
            "num_children",
            "phone_ownership",
        ]
        actually_updated_fields = {}
        for k, v in extracted_data_from_pipeline.items():
            if k in onboarding_data_to_collect:
                if updated_user_context.get(k) != v:
                    logger.info(
                        f"Updating user_context for {k}: from "
                        f"'{updated_user_context.get(k)}' to '{v}'"
                    )
                    updated_user_context[k] = v
                    actually_updated_fields[k] = v
            else:
                if updated_user_context.get("other", {}).get(k) != v:
                    logger.info(
                        f"Updating user_context['other'] for {k}: '{v}'"
                    )
                    if "other" not in updated_user_context:
                        updated_user_context["other"] = {}
                    updated_user_context["other"][k] = v
                    if "other" not in actually_updated_fields:
                        actually_updated_fields["other"] = {}
                    actually_updated_fields["other"][k] = v

        if actually_updated_fields:
            logger.info(
                "Fields updated in user_context: "
                f"{json.dumps(actually_updated_fields)}"
            )
            try:
                current_observation = observe.get_current_observation()
                if current_observation:
                    existing_metadata = current_observation.metadata or {}
                    current_observation.metadata(
                        {
                            **existing_metadata,
                            "applied_extracted_data": actually_updated_fields,
                        }
                    )
            except Exception as e:
                logger.warning(
                    "Could not update Langfuse observation metadata "
                    f"with applied data: {e}"
                )
    else:
        logger.warning(
            "Data extraction pipeline did not produce a result "
            "(extracted_data_from_pipeline is empty/None)."
        )

    return updated_user_context


@observe()
def get_assessment_question(
    flow_id: str,
    question_number: int,
    current_assessment_step: int,
    user_context: dict,
    question_to_contextualize: str = None,
) -> dict:
    """
    Gets and contextualizes a specific assessment question.
    """
    if current_assessment_step >= len(all_assessment_questions):
        logger.info("Assessment flow complete.")
        return {}

    next_step_data = all_assessment_questions[current_assessment_step]
    actual_question_number_from_flow = next_step_data["question_number"]
    raw_question_content = next_step_data["content"]

    if actual_question_number_from_flow != question_number:
        logger.warning(
            f"Provided question_number {question_number} does not match "
            f"flow's current step question_number "
            f"{actual_question_number_from_flow}."
        )

    logger.info(
        f"Processing step {actual_question_number_from_flow} for flow "
        f"'{ASSESSMENT_FLOW_ID}'. Original content: '{raw_question_content}'"
    )

    logger.info("Running contextualization pipeline...")
    assessment_contextualization_pipe = (
        pipelines.create_assessment_contextualization_pipeline()
    )
    contextualized_question_from_pipeline = (
        pipelines.run_assessment_contextualization_pipeline(
            assessment_contextualization_pipe,
            ASSESSMENT_FLOW_ID,  # Using constant flow_id for consistency
            actual_question_number_from_flow,
            user_context,
            raw_question_content,  # Explicitly passing raw question
        )
    )

    final_contextualized_question = contextualized_question_from_pipeline
    if not contextualized_question_from_pipeline:
        logger.warning(
            f"Contextualization failed for step "
            f"{actual_question_number_from_flow}. "
            "Using raw content as fallback."
        )
        final_contextualized_question = raw_question_content

    try:
        current_observation = observe.get_current_observation()
        if current_observation:
            current_observation.metadata(
                {
                    "assessment_flow_id": ASSESSMENT_FLOW_ID,
                    "question_number_from_flow": (
                        actual_question_number_from_flow
                    ),
                    "raw_question_content": raw_question_content,
                    "contextualization_pipeline_output": (
                        contextualized_question_from_pipeline
                    ),
                    "used_fallback": not bool(
                        contextualized_question_from_pipeline
                    ),
                }
            )
    except Exception as e:
        logger.warning(f"Could not update Langfuse observation metadata: {e}")

    return {
        "contextualized_question": final_contextualized_question,
        "current_question_number": actual_question_number_from_flow,
    }


@observe()
def validate_assessment_answer(
    user_response: str,
    current_question_number: int,
    valid_responses_options: list = None,
) -> dict:
    """
    Validates a user's response to an assessment question.
    """
    validator_pipe = pipelines.create_assessment_response_validator_pipeline()
    processed_user_response_from_pipeline = (
        pipelines.run_assessment_response_validator_pipeline(
            validator_pipe,
            user_response,
            current_question_number,
            valid_responses_options,
        )
    )

    if not processed_user_response_from_pipeline:
        logger.warning(
            "Response validation pipeline failed to produce a processed "
            f"response for question {current_question_number}."
        )
    else:
        logger.info(
            f"Storing validated response for question_number "
            f"{current_question_number}: "
            f"{processed_user_response_from_pipeline}"
        )

    try:
        current_observation = observe.get_current_observation()
        if current_observation:
            current_observation.metadata(
                {
                    "processed_response_from_pipeline": (
                        processed_user_response_from_pipeline
                    ),
                    "related_question_number": current_question_number,
                    "provided_valid_options_for_validation": (
                        valid_responses_options
                    ),
                }
            )
    except Exception as e:
        logger.warning(f"Could not update Langfuse observation metadata: {e}")

    return {
        "processed_user_response": processed_user_response_from_pipeline,
        "current_assessment_step": (
            current_question_number
        ),  # This validation pertains to this step
    }
