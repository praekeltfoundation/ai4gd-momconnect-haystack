import logging
import json
from typing import Any
from pydantic import ValidationError


from . import doc_store
from . import pipelines
from .models import AssessmentRun, Question, Turn

# --- Configuration ---
logger = logging.getLogger(__name__)

onboarding_flow_id = "onboarding"
dma_flow_id = "dma-assessment"

all_onboarding_questions = doc_store.ONBOARDING_FLOW.get(onboarding_flow_id, [])
all_dma_questions = doc_store.DMA_FLOW.get(dma_flow_id, [])


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
    if current_assessment_step >= len(all_dma_questions):
        logger.info("Assessment flow complete.")
        return {}

    next_step_data = all_dma_questions[current_assessment_step]
    current_question_number = next_step_data["question_number"]
    logger.info(f"Processing step {current_question_number} for flow '{dma_flow_id}'")

    # Contextualize the current question
    logger.info("Running contextualization pipeline...")
    # Call with correct arguments: pipeline, flow_id, question_number, user_context
    assessment_contextualization_pipe = (
        pipelines.create_assessment_contextualization_pipeline()
    )
    contextualized_question = pipelines.run_assessment_contextualization_pipeline(
        assessment_contextualization_pipe,
        dma_flow_id,
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


def load_and_validate_assessment_questions(
    assessment_id: str, assessment_data_source: dict
) -> list[Question] | None:
    """
    Extracts and validates a specific list of questions from a larger
    assessment data source (like the KAB or DMA dictionary).
    """
    logger.info(f"Loading and validating questions for '{assessment_id}'...")
    raw_question_data = assessment_data_source.get(assessment_id)

    if not raw_question_data or not isinstance(raw_question_data, list):
        logger.error(
            f"No valid question data found for '{assessment_id}' in the provided source."
        )
        return None

    try:
        validated_questions = [Question.model_validate(q) for q in raw_question_data]
        logger.info(
            f"Successfully validated {len(validated_questions)} questions for '{assessment_id}'."
        )
        return validated_questions
    except ValidationError as e:
        logger.critical(f"Data structure error in source for '{assessment_id}':\n{e}")
        return None


def _calculate_assessment_score_range(
    assessment_questions: list[Question],
) -> tuple[int, int]:
    """
    Calculates the min/max possible scores using validated Question models.
    """
    total_min_score, total_max_score = 0, 0
    for question in assessment_questions:
        if scores := list(question.valid_responses.values()):
            total_min_score += min(scores)
            total_max_score += max(scores)
    return total_min_score, total_max_score


def _score_single_turn(
    turn: Turn,
    question_lookup_by_num: dict[int, Question],
) -> dict[str, Any]:
    """
    Scores a single assessment turn.
    """
    # This function now safely assumes turn.question_number is not None
    # because the calling function is responsible for filtering.
    q_num = turn.question_number
    question_data = question_lookup_by_num.get(q_num)
    result = turn.model_dump()

    result["score"] = 0
    if not question_data:
        result["score_error"] = (
            f"Question number {q_num} not found in master assessment file."
        )
        return result

    # Add rich details to the result
    # result["question_name"] = question_data.question_name
    # result["question_text"] = question_data.content

    # Calculate the score
    if isinstance(question_data.valid_responses, dict):
        score_val = question_data.valid_responses.get(turn.user_response)
        if score_val is not None:
            result["score"] = score_val
        else:
            result["score_error"] = "User's answer is not a valid, scorable option."
    else:
        result["score_error"] = "This question is not structured for scoring."

    return result


def score_assessment_from_simulation(
    simulation_output: list[AssessmentRun],
    assessment_id: str,
    assessment_questions: list[Question],
) -> dict[str, Any] | None:
    """
    Calculates the final score for a specific assessment run.
    """
    logger.info(
        f"Calculating final score for '{assessment_id}' from simulation output..."
    )

    assessment_run = next(
        (run for run in simulation_output if run.flow_type == assessment_id),
        None,
    )
    if not assessment_run:
        logger.error(
            f"Could not find a valid run for '{assessment_id}' in the simulation output."
        )
        return None

    question_lookup = {q.question_number: q for q in assessment_questions}

    # ---Score Calculation Logic ---
    min_possible_score, max_possible_score = _calculate_assessment_score_range(
        assessment_questions
    )
    results = [
        _score_single_turn(turn, question_lookup)
        for turn in assessment_run.turns
        if turn.question_number is not None
    ]
    user_total_score = sum(r["score"] for r in results if r.get("score") is not None)
    score_percentage = (
        (user_total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
    )

    logger.info(f"Final score for '{assessment_id}' calculated: {user_total_score}")

    return {
        "overall_score": user_total_score,
        "score_percentage": round(score_percentage, 2),
        "assessment_min_score": min_possible_score,
        "assessment_max_score": max_possible_score,
        "results": results,
    }
