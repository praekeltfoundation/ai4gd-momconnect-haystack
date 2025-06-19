import json
import logging
from typing import Any

from haystack.dataclasses import ChatMessage
from pydantic import ValidationError

from . import doc_store, pipelines
from .pydantic_models import AssessmentRun, Question, Turn

# --- Configuration ---
logger = logging.getLogger(__name__)

onboarding_flow_id = "onboarding"
dma_flow_id = "dma-assessment"
kab_k_flow_id = "knowledge-assessment"
kab_a_flow_id = "attitude-assessment"
kab_b_pre_flow_id = "behaviour-pre-assessment"
kab_b_post_flow_id = "behaviour-post-assessment"
anc_survey_flow_id = "anc-survey"
faqs_flow_id = "faqs"

all_onboarding_questions = doc_store.ONBOARDING_FLOW.get(onboarding_flow_id, [])
all_dma_questions = doc_store.DMA_FLOW.get(dma_flow_id, [])
all_kab_k_questions = doc_store.KAB_FLOW.get(kab_k_flow_id, [])
all_kab_a_questions = doc_store.KAB_FLOW.get(kab_a_flow_id, [])
all_kab_b_pre_questions = doc_store.KAB_FLOW.get(kab_b_pre_flow_id, [])
all_kab_b_post_questions = doc_store.KAB_FLOW.get(kab_b_post_flow_id, [])
all_anc_survey_questions = doc_store.ANC_SURVEY_FLOW.get(anc_survey_flow_id, [])
faq_questions = doc_store.FAQ_DATA.get(faqs_flow_id, [])

assessment_flow_map = {
    dma_flow_id: all_dma_questions,
    kab_k_flow_id: all_kab_k_questions,
    kab_a_flow_id: all_kab_a_questions,
    kab_b_pre_flow_id: all_kab_b_pre_questions,
    kab_b_post_flow_id: all_kab_b_post_questions,
}

ANC_SURVEY_MAP = {item["title"]: item for item in all_anc_survey_questions}


def get_next_onboarding_question(
    user_context: dict, chat_history: list[ChatMessage]
) -> str | None:
    """
    Gets the next contextualized onboarding question.
    """
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
    user_response: str, user_context: dict, chat_history: list[ChatMessage]
) -> dict:
    """
    Extracts data from a user's response to an onboarding question.
    """
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
                user_context.setdefault("other", {})[k] = v
    else:
        logger.warning("Data extraction pipeline did not produce a result.")
    return user_context


def get_assessment_question(
    flow_id: str, question_number: int, user_context: dict
) -> dict:
    """
    Gets the next contextualized assessment question from the specified flow.

    question_number uses 1-based indexing
    """
    # Get the list of questions for the specified flow_id
    question_list = assessment_flow_map.get(flow_id)
    if not question_list:
        logger.error(f"Invalid flow_id provided: '{flow_id}'. No questions found.")
        return {}

    if question_number > len(question_list):
        logger.info(f"Assessment flow '{flow_id}' complete.")
        return {}

    # Contextualize the question
    logger.info("Running contextualization pipeline...")
    assessment_contextualization_pipe = (
        pipelines.create_assessment_contextualization_pipeline()
    )
    contextualized_question = pipelines.run_assessment_contextualization_pipeline(
        assessment_contextualization_pipe,
        flow_id,
        question_number,
        user_context,
    )

    if not contextualized_question:
        logger.error(f"Question contextualization failed in flow: '{flow_id}'.")
        return {}

    return {
        "contextualized_question": contextualized_question,
    }


def validate_assessment_answer(
    user_response: str, question_number: int, current_flow_id: str
) -> dict[str, Any]:
    """
    Validates a user's response to an assessment question.

    question_number uses 1-based indexing
    """
    current_question = None
    for q in assessment_flow_map[current_flow_id]:
        if q["question_number"] == question_number:
            current_question = q
    if not current_question:
        logger.error(
            f"Could not find question number {question_number} in flow '{current_flow_id}'."
        )
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }
    valid_responses_and_scores = current_question.get("valid_responses_and_scores", [])
    if not valid_responses_and_scores:
        logger.error(f"No valid responses found for question {question_number}.")
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }
    valid_responses = [
        item["response"] for item in valid_responses_and_scores if "response" in item
    ]

    validator_pipe = pipelines.create_assessment_response_validator_pipeline()
    processed_user_response = pipelines.run_assessment_response_validator_pipeline(
        validator_pipe, user_response, valid_responses
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


def get_anc_survey_question(
    user_context: dict, chat_history: list[ChatMessage]
) -> dict | None:
    """
    Gets the next contextualized ANC survey question by first determining the
    next logical step, then fetching the content and contextualizing it.
    """
    # 1. Get the next logical step from the Navigator Pipeline
    logger.info("Running clinic visit navigator pipeline...")
    navigator_pipe = pipelines.create_clinic_visit_navigator_pipeline()
    if not navigator_pipe:
        logger.error("Failed to create ANC survey navigator pipeline.")
        return None

    step_result = pipelines.run_clinic_visit_navigator_pipeline(
        navigator_pipe, user_context
    )

    if not step_result or "next_step" not in step_result:
        logger.error("Question navigation failed in 'anc-survey' flow.")
        return None

    next_step_id = step_result["next_step"]
    is_final = step_result.get("is_final_step", False)

    # 2. Fetch the original question content from our loaded JSON data
    question_data = ANC_SURVEY_MAP.get(next_step_id)
    if not question_data:
        logger.error(f"Could not find question content for step_id: '{next_step_id}'")
        return None

    original_question_content = question_data.get("content", "")
    valid_responses = question_data.get("valid_respnses", [])

    # 3. Get the final, contextualized question from the Contextualizer Pipeline
    logger.info(f"Running contextualization pipeline for step: '{next_step_id}'...")
    contextualizer_pipe = pipelines.create_anc_survey_contextualization_pipeline()
    if not contextualizer_pipe:
        logger.error("Failed to create ANC survey contextualization pipeline.")
        # Fallback to original content if pipeline creation fails
        contextualized_question = original_question_content
    else:
        contextualized_question = pipelines.run_anc_survey_contextualization_pipeline(
            contextualizer_pipe,
            user_context,
            chat_history,
            original_question_content,
            valid_responses,
        )

    # 4. Append valid responses to the final question, if they exist
    valid_responses = question_data.get("valid_responses")
    if valid_responses:
        options = "\n".join(valid_responses)
        final_question_text = f"{contextualized_question}\n\n{options}"
    else:
        final_question_text = contextualized_question

    return {
        "contextualized_question": final_question_text.strip(),
        "is_final_step": is_final,
    }


def extract_anc_data_from_response(
    user_response: str, user_context: dict, chat_history: list[ChatMessage]
) -> dict:
    """
    Extracts data from a user's response to an ANC survey question.
    """
    logger.info("Running ANC survey data extraction pipeline...")
    anc_data_extraction_pipe = pipelines.create_clinic_visit_data_extraction_pipeline()
    extracted_data = pipelines.run_clinic_visit_data_extraction_pipeline(
        anc_data_extraction_pipe, user_response, user_context, chat_history
    )

    print(f"[Extracted ANC Data]:\n{json.dumps(extracted_data, indent=2)}\n")

    if extracted_data:
        # Update the user_context with the newly extracted data
        for key, value in extracted_data.items():
            user_context[key] = value
            logger.info(f"Updated user_context for {key}: {value}")
    else:
        logger.warning("ANC data extraction pipeline did not produce a result.")

    return user_context


def detect_user_intent(last_question: str, user_response: str) -> str | None:
    """Runs the intent detection pipeline."""
    logger.info("Running intent detection pipeline...")
    intent_pipeline = pipelines.create_intent_detection_pipeline()
    result = pipelines.run_intent_detection_pipeline(
        intent_pipeline, last_question, user_response
    )
    return result.get("intent") if result else None


def get_faq_answer(user_question: str) -> str | None:
    """Runs the FAQ answering pipeline."""
    logger.info("Running FAQ answering pipeline...")
    faq_pipeline = pipelines.create_faq_answering_pipeline()
    result = pipelines.run_faq_pipeline(
        faq_pipeline,
        user_question,
        filters={"field": "meta.flow_id", "operator": "==", "value": "faqs"},
    )
    return result.get("answer") if result else None


def handle_user_message(
    previous_question: str, user_message: str
) -> tuple[str | None, str | None]:
    """
    Orchestrates the process of handling a new user message.
    """
    intent = detect_user_intent(previous_question, user_message)
    response = None

    if intent == "QUESTION_ABOUT_STUDY":
        response = get_faq_answer(user_question=user_message)
        if not response:
            response = "Sorry, I don't have information about that right now."
    elif intent == "HEALTH_QUESTION":
        response = "Please consider first finishing the current survey, but we may have an answer to your question on MomConnect: https://wa.me/27796312456?text=menu"
    elif intent == "CHITCHAT":
        if previous_question and previous_question != "":
            response = "Please try answering the previous question again."
        else:
            response = "Thank you for reaching out! We will let you know if we have more questions for you. You can also click on this link to go to MomConnect: https://wa.me/27796312456?text=menu"
    elif intent in [
        "JOURNEY_RESPONSE",
        "ASKING_TO_STOP_MESSAGES",
        "ASKING_TO_DELETE_DATA",
        "REPORTING_AIRTIME_NOT_RECEIVED",
    ]:
        pass
    else:
        logger.error(f"Intent detected: {intent}. No specific action defined.")

    return intent, response


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
        if isinstance(question.valid_responses, dict):
            scores = list(question.valid_responses.values())
            if scores:
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
        "turns": results,
    }
