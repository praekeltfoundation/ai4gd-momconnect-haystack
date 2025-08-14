import logging
from typing import Any
from pydantic import ValidationError

from ai4gd_momconnect_haystack import pipelines
from ai4gd_momconnect_haystack.enums import AssessmentType
from ai4gd_momconnect_haystack.pydantic_models import (
    AssessmentEndContentItem,
    AssessmentEndResponse,
    AssessmentEndScoreBasedMessage,
    AssessmentEndSimpleMessage,
    AssessmentQuestion,
    AssessmentResult,
    AssessmentRun,
    ResponseScore,
    Turn,
)
from ai4gd_momconnect_haystack.utilities import (
    assessment_map_to_their_pre,
    assessment_flow_map,
    dma_post_flow_id,
    dma_pre_flow_id,
    kab_a_post_flow_id,
    kab_a_pre_flow_id,
    kab_b_post_flow_id,
    kab_k_post_flow_id,
    kab_k_pre_flow_id,
    prepare_valid_responses_to_use_in_assessment_system_prompt,
)


logger = logging.getLogger(__name__)


def load_and_validate_assessment_questions(
    assessment_id: str,
) -> list[AssessmentQuestion] | None:
    """
    Extracts and validates a specific list of questions using the pre-loaded
    assessment_flow_map.
    """
    logger.info(f"Loading and validating questions for '{assessment_id}'...")

    # Use the existing global map to get the raw question data
    raw_question_data = assessment_flow_map.get(assessment_id)

    if not raw_question_data or not isinstance(raw_question_data, list):
        logger.error(
            f"No valid question data found for '{assessment_id}' in the assessment_flow_map."
        )
        return None

    try:
        validated_questions = [
            AssessmentQuestion.model_validate(q) for q in raw_question_data
        ]
        logger.info(
            f"Successfully validated {len(validated_questions)} questions for '{assessment_id}'."
        )
        return validated_questions
    except ValidationError as e:
        logger.critical(f"Data structure error in source for '{assessment_id}':\n{e}")
        return None


def _calculate_assessment_score_range(
    assessment_questions: list[AssessmentQuestion],
) -> tuple[int, int]:
    """
    Calculates the min/max possible scores using validated Question models.
    """
    total_min_score, total_max_score = 0, 0
    for question in assessment_questions:
        # Assumes the Pydantic model now has 'valid_responses_and_scores'
        responses_and_scores = question.valid_responses_and_scores
        if isinstance(responses_and_scores, list):
            scores = [
                item.score
                for item in responses_and_scores
                if isinstance(item, ResponseScore) and item.score is not None
            ]
        else:
            scores = []

        if scores:
            total_min_score += min(scores)
            total_max_score += max(scores)
    return total_min_score, total_max_score


def _score_single_turn(
    turn: Turn,
    question_lookup_by_num: dict[int, AssessmentQuestion],
) -> dict[str, Any]:
    """
    Scores a single assessment turn.
    """
    result = turn.model_dump()
    result["score"] = 0

    q_num = turn.question_number
    if q_num is None:
        result["score_error"] = "Turn has no question number."
        return result

    question_data = question_lookup_by_num.get(q_num)
    if not question_data:
        result["score_error"] = (
            f"Question number {q_num} not found in master assessment file."
        )
        return result

    responses_and_scores = getattr(question_data, "valid_responses_and_scores", None)
    if isinstance(responses_and_scores, list) and all(
        isinstance(i, ResponseScore) for i in responses_and_scores
    ):
        score_val = None
        # Find the matching response in the list of ResponseScore objects
        for item in responses_and_scores:
            if isinstance(item, ResponseScore) and item.response == turn.user_response:
                score_val = item.score
                break  # Found a match

        if score_val is not None:
            result["score"] = score_val
        else:
            result["score_error"] = "User's answer is not a valid, scorable option."
    else:
        result["score_error"] = "This question is not structured for scoring."

    return result


def validate_assessment_answer(
    user_response: str, question_number: int, current_flow_id: str
) -> dict[str, Any]:
    """
    Validates a user's response to an assessment question.

    question_number uses 1-based indexing
    """
    current_question = None
    for q in assessment_flow_map[current_flow_id]:
        if q.question_number == question_number:
            current_question = q
    if not current_question:
        logger.error(
            f"Could not find question number {question_number} in flow '{current_flow_id}'."
        )
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }
    valid_responses_and_scores = current_question.valid_responses_and_scores
    if not valid_responses_and_scores:
        logger.error(f"No valid responses found for question {question_number}.")
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }
    valid_responses = [item.response for item in valid_responses_and_scores]
    if "dma" in current_flow_id:
        flow_id_to_use = "dma-assessment"
    elif "knowledge" in current_flow_id:
        flow_id_to_use = "knowledge-assessment"
    elif "attitude" in current_flow_id:
        flow_id_to_use = "attitude-assessment"
    else:
        flow_id_to_use = "behaviour-pre-assessment"
    valid_responses_for_prompt = (
        prepare_valid_responses_to_use_in_assessment_system_prompt(
            flow_id_to_use, question_number, current_question
        )
    )

    processed_user_response = pipelines.run_assessment_response_validator_pipeline(
        user_response, valid_responses, valid_responses_for_prompt
    )

    # Move to the next step, or try again if the response was invalid
    if processed_user_response:
        logger.info(
            f"Storing validated response for question_number {question_number}: {processed_user_response}"
        )
        question_number += 1
    else:
        logger.warning(f"Response validation failed for step {question_number}.")

    return {
        "processed_user_response": processed_user_response,
        "next_question_number": question_number,
    }


def validate_assessment_end_response(
    previous_message: str,
    previous_message_nr: int,
    previous_message_valid_responses: list[str],
    user_response: str,
) -> dict[str, Any]:
    """ """
    # Normalize the user's input before sending it to the pipeline
    normalized_user_response = user_response.lower().strip()

    # Create and run a pipeline that validates the user's response to the previous message
    processed_user_response = pipelines.run_assessment_end_response_validator_pipeline(
        normalized_user_response,
        previous_message_valid_responses,
        previous_message,
    )
    # Move to the next step, or try again if the response was invalid
    if processed_user_response:
        logger.info(
            f"Storing validated response for message_number {previous_message_nr}: {processed_user_response}"
        )
        next_message_nr = previous_message_nr + 1
    else:
        logger.warning(f"Response validation failed for step {previous_message_nr}.")
        next_message_nr = previous_message_nr

    return {
        "processed_user_response": processed_user_response,
        "next_message_number": next_message_nr,
    }


def score_assessment_from_simulation(
    simulation_output: list[AssessmentRun],
    assessment_id: str,
    assessment_questions: list[AssessmentQuestion],
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
        "turns": results,
    }


def score_assessment_question(
    user_response: str, question_number: int, flow_id: AssessmentType
) -> int | None:
    assessment_questions = load_and_validate_assessment_questions(
        assessment_map_to_their_pre[flow_id.value]
    )
    if assessment_questions:
        question_lookup = {q.question_number: q for q in assessment_questions}
        score_result = _score_single_turn(
            Turn.model_validate(
                {"user_response": user_response, "question_number": question_number}
            ),
            question_lookup,
        )
        if "score" in score_result.keys():
            return score_result["score"]
    return None


def score_assessment(
    assessment_run: AssessmentRun,
    assessment_id: AssessmentType,
) -> AssessmentResult | None:
    assessment_questions = load_and_validate_assessment_questions(
        assessment_map_to_their_pre[assessment_id.value]
    )
    if not assessment_questions:
        logger.error(
            "Function 'load_and_validate_assessment_questions' called incorrectly"
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
    skip_count = sum(turn.user_response == "Skip" for turn in assessment_run.turns)
    category = ""
    crossed_skip_threshold = False
    if assessment_id.value in [dma_pre_flow_id, dma_post_flow_id]:
        if score_percentage > 67:
            category = "high"
        elif score_percentage > 33:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    elif assessment_id.value in [kab_k_pre_flow_id, kab_k_post_flow_id]:
        if score_percentage > 83:
            category = "high"
        elif score_percentage > 50:
            category = "medium"
        else:
            category = "low"
        if skip_count > 3:
            crossed_skip_threshold = True
    elif assessment_id.value in [kab_a_pre_flow_id, kab_a_post_flow_id]:
        if score_percentage >= 80:
            category = "high"
        elif score_percentage >= 60:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    else:
        if score_percentage == 100:
            category = "high"
        elif score_percentage >= 75:
            category = "medium"
        else:
            category = "low"
        if skip_count > 2:
            crossed_skip_threshold = True
    return AssessmentResult.model_validate(
        {
            "score": score_percentage,
            "category": category,
            "crossed_skip_threshold": crossed_skip_threshold,
        }
    )


def get_content_from_message_data(
    message_data: AssessmentEndScoreBasedMessage | AssessmentEndSimpleMessage,
    score_category: str,
) -> tuple[str, list[str] | None]:
    """
    Extracts the content and valid responses from a message object based on the score category.
    """
    if isinstance(message_data, AssessmentEndSimpleMessage):
        return message_data.content, message_data.valid_responses

    if isinstance(message_data, AssessmentEndScoreBasedMessage):
        content_map = {
            "high": message_data.high_score_content,
            "medium": message_data.medium_score_content,
            "low": message_data.low_score_content,
            "skipped-many": message_data.skipped_many_content,
        }
        # Get the specific content block for the score
        score_content = content_map.get(
            score_category,
            AssessmentEndContentItem.model_validate(
                {"content": "", "valid_responses": []}
            ),
        )
        return score_content.content, score_content.valid_responses

    # This should ideally not be reached if type checking is correct
    raise TypeError("Unsupported message data type")


def determine_task(
    flow_id: str,
    previous_message_nr: int,
    score_category: str,
    processed_user_response: str,
) -> str:
    """
    Refactored Logic: Encapsulates the business logic for determining
    the background task.
    """
    if flow_id == "dma-pre-assessment":
        if (
            previous_message_nr == 1
            and score_category == "skipped-many"
            and processed_user_response == "Yes"
        ):
            return "REMIND_ME_LATER"
        if previous_message_nr == 2:
            return "STORE_FEEDBACK"

    if flow_id in ["behaviour-pre-assessment", "knowledge-pre-assessment"]:
        if previous_message_nr == 1 and processed_user_response == "Remind me tomorrow":
            return "REMIND_ME_LATER"

    if flow_id == "attitude-pre-assessment" and previous_message_nr == 1:
        if score_category == "skipped-many" and processed_user_response == "Yes":
            return "REMIND_ME_LATER"
        elif score_category != "skipped-many":
            return "STORE_FEEDBACK"

    return ""


def matches_assessment_question_length(
    n_questions: int, assessment_type: AssessmentType
):
    """
    Checks if the length of a list of AssessmentHistory objects matches the length of questions from the corresponding assessment.
    """
    if assessment_type.value == kab_b_post_flow_id:
        assessment_type = assessment_map_to_their_pre[kab_b_post_flow_id]
    if len(assessment_flow_map[assessment_type.value]) == n_questions:
        return True
    return False


def create_assessment_end_error_response(reason: str) -> AssessmentEndResponse:
    logger.warning(reason)
    return AssessmentEndResponse(
        message="",
        task="ERROR",
        intent="",
        intent_related_response="",
    )


def response_is_required_for(flow_id: str, message_nr: int) -> bool:
    required_map = {
        "dma-pre-assessment": [2, 3],
        "behaviour-pre-assessment": [2],
        "knowledge-pre-assessment": [2],
        "attitude-pre-assessment": [2],
    }
    return message_nr in required_map.get(flow_id, [])
