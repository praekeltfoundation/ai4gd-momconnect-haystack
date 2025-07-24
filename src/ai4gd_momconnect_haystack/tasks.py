import json
import logging


from ai4gd_momconnect_haystack.crud import (
    get_assessment_history,
    get_or_create_chat_history,
)
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType
from ai4gd_momconnect_haystack.pipelines import (
    get_next_anc_survey_step,
    run_anc_survey_contextualization_pipeline,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import AssessmentHistory

from . import doc_store, pipelines
from .utilities import (
    all_onboarding_questions,
    ANC_SURVEY_MAP,
    assessment_flow_map,
    assessment_map_to_their_pre,
    kab_b_post_flow_id,
    kab_b_pre_flow_id,
    prepare_valid_responses_to_display_to_anc_survey_user,
    prepare_valid_responses_to_display_to_assessment_user,
    prepare_valid_responses_to_display_to_onboarding_user,
    prepare_valid_responses_to_use_in_anc_survey_system_prompt,
)


# --- Configuration ---
logger = logging.getLogger(__name__)


def get_next_onboarding_question(
    user_context: dict, append_valid_responses: bool = False
) -> dict | None:
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

    # Determine first/last question flags
    is_first_question = len(remaining_questions_list) == len(all_onboarding_questions)
    is_last_question = len(remaining_questions_list) == 1

    # LLM decides the next question
    logger.info("Running next question selection pipeline...")
    next_question_result = pipelines.run_next_onboarding_question_pipeline(
        user_context,
        [q.model_dump() for q in remaining_questions_list],
        is_first_question=is_first_question,
        is_last_question=is_last_question,
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
            if q.question_number == chosen_question_number
        ),
        None,
    )

    if not current_step_data:
        logger.error(
            f"Could not find question data for question_number {chosen_question_number}. Skipping."
        )
        return None
    else:
        logger.info(f"Original question was: '{current_step_data.content}'")

    # Append valid responses to the final question, when applicable
    valid_responses = current_step_data.valid_responses
    if append_valid_responses and valid_responses and current_step_data.collects:
        final_question_text = prepare_valid_responses_to_display_to_onboarding_user(
            contextualized_question, current_step_data.collects, valid_responses
        )
    else:
        final_question_text = contextualized_question

    return {
        "contextualized_question": final_question_text,
        "question_number": chosen_question_number,
    }


def extract_onboarding_data_from_response(
    user_response: str,
    user_context: dict,
    current_question: str,
) -> dict:
    """
    Extracts data from a user's response to an onboarding question
    and returns ONLY the new data dictionary.
    """
    logger.info("Running data extraction pipeline...")
    extracted_data = pipelines.run_onboarding_data_extraction_pipeline(
        user_response,
        user_context,
        current_question,
    )

    if extracted_data:
        print(f"[Extracted Data]:\n{json.dumps(extracted_data, indent=2)}\n")
        return extracted_data  # Return only the new dictionary of updates.

    logger.warning("Data extraction pipeline did not produce a result.")
    return {}


def update_context_from_onboarding_response(
    user_input: str, current_context: dict, current_question: str
) -> dict:
    """
    Takes user input, extracts data, and returns the fully updated context.
    This is the core business logic for an onboarding turn.
    """
    updated_context = current_context.copy()

    updates = extract_onboarding_data_from_response(
        user_response=user_input,
        user_context=current_context,
        current_question=current_question,
    )

    if updates:
        onboarding_data_to_collect = [
            q.collects for q in all_onboarding_questions if q.collects
        ]
        for key, value in updates.items():
            if key in onboarding_data_to_collect:
                updated_context[key] = value
            else:
                updated_context.setdefault("other", {})[key] = value

    return updated_context


def process_onboarding_step(
    user_input: str, current_context: dict, current_question: str
) -> tuple[dict, dict | None]:
    """
    Processes a single step of the onboarding flow for the API.
    """
    updated_context = update_context_from_onboarding_response(
        user_input,
        current_context,
        current_question,
    )

    next_question = get_next_onboarding_question(user_context=updated_context)

    return updated_context, next_question


async def get_assessment_question(
    user_id: str, flow_id: AssessmentType, question_number: int, user_context: dict
) -> dict:
    """
    Gets the next contextualized assessment question from the specified flow.

    question_number uses 1-based indexing
    """
    # Check if the requested question has already been processed and stored in
    # the pre-assessment history. This might happen if the user dropped off
    # and is now returning to continue.
    pre_assessment_flow_id = assessment_map_to_their_pre[flow_id.value]
    question_history: list[AssessmentHistory] = await get_assessment_history(
        user_id, pre_assessment_flow_id
    )
    if question_history:
        for q in question_history:
            if q.question_number == question_number:
                # Whether the user is doing pre- or post-, we return the question
                # unless it's the special case of the KAB Behaviour post-assessment,
                # in which case we need to generate a new question.
                if flow_id.value == kab_b_post_flow_id and question_number == len(
                    assessment_flow_map[kab_b_pre_flow_id]
                ):
                    logger.info("Running contextualization pipeline...")
                    contextualized_question = (
                        pipelines.run_assessment_contextualization_pipeline(
                            flow_id.value,
                            question_number,
                            user_context,
                        )
                    )
                    if not contextualized_question:
                        logger.error(
                            f"Question contextualization failed in flow: '{flow_id.value}'."
                        )
                        return {}
                    return {
                        "contextualized_question": contextualized_question,
                    }
                else:
                    return {
                        "contextualized_question": q.question,
                    }
    # If
    # - the question history is empty, or
    # - the question number is not in the history, or
    # - the question number is in the history but it's not the special case of the KAB Behaviour post-assessment,
    # then we proceed to generate the requested question.

    # For the DMA, KAB Knowledge and KAB Attitude flows, we use the same pre
    # and post flows.
    # For the KAB Behaviour flow, we use the same questions for pre and post
    # except for the last question, which is different. For this assessment,
    # we first use the pre-assessment flow to get all the questions and then
    # we replace the last question with the post-assessment question.

    if "dma" in flow_id.value:
        flow_id_to_use = "dma-assessment"
    elif "knowledge" in flow_id.value:
        flow_id_to_use = "knowledge-assessment"
    elif "attitude" in flow_id.value:
        flow_id_to_use = "attitude-assessment"
    else:
        flow_id_to_use = "behaviour-pre-assessment"

    question_list = assessment_flow_map.get(flow_id.value)
    if not question_list:
        logger.error(f"Invalid flow_id: '{flow_id.value}'. No questions found.")
        return {}

    if question_number > len(question_list):
        logger.info(
            f"Question number '{question_number}' for flow '{flow_id.value}' does not exist. End of flow."
        )
        return {}

    # Contextualize the question
    logger.info("Running contextualization pipeline...")
    contextualized_question = pipelines.run_assessment_contextualization_pipeline(
        flow_id_to_use,
        question_number,
        user_context,
    )

    if not contextualized_question:
        logger.error(f"Question contextualization failed in flow: '{flow_id.value}'.")
        return {}

    question_data = [q for q in question_list if q.question_number == question_number][
        -1
    ]

    # For KAB Behaviour assessments, the user provides free-text input without seeing options.
    # For DMA, KAB Knowledge, and KAB Attitude, we display the options.
    if "behaviour" not in flow_id.value:
        contextualized_question = prepare_valid_responses_to_display_to_assessment_user(
            flow_id_to_use, question_number, contextualized_question, question_data
        )

    return {
        "contextualized_question": contextualized_question,
    }


def extract_assessment_data_from_response(
    user_response: str,
    flow_id: str,
    question_number: int,
) -> dict:  # Changed return type
    """
    Extracts data from a user's response to an assessment question and
    returns a dictionary with the processed response and next question number.
    """
    logger.info("Running assessment data extraction pipeline...")

    # 1. Get the question data for the current step
    question_list = assessment_flow_map.get(flow_id)
    if not question_list:
        logger.error(f"Invalid flow_id: '{flow_id}'. No questions found.")
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }

    try:
        question_data = [
            q for q in question_list if q.question_number == question_number
        ][-1]
    except IndexError:
        logger.error(
            f"Could not find question data for question_number {question_number} in flow {flow_id}."
        )
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }

    # 2. Get the clean list of valid responses for the final schema check
    valid_responses = []
    if question_data.valid_responses_and_scores:
        valid_responses = [
            item.response
            for item in question_data.valid_responses_and_scores
            if item.response != "Skip"
        ]

    # 3. Re-create the full message that was sent to the user
    extracted_data = None
    if "behaviour" in flow_id:
        # For KAB-B, use a pipeline that maps free text to unseen valid responses,
        # similar to the ANC survey. The user only sees the question content.
        # We reuse the clinic visit extraction pipeline as it's designed for this.
        extracted_data = pipelines.run_behaviour_data_extraction_pipeline(
            user_response=user_response,
            previous_service_message=question_data.content or "",
            valid_responses=valid_responses,
        )
    else:
        # For DMA, KAB-K, KAB-A, the user sees the options.
        # Re-create the full message sent to the user (question + options)
        previous_message = prepare_valid_responses_to_display_to_assessment_user(
            flow_id, question_number, question_data.content or "", question_data
        )
        # 4. Call the new, reliable extraction pipeline
        extracted_data = pipelines.run_assessment_data_extraction_pipeline(
            user_response=user_response,
            previous_message=previous_message,
            valid_responses=valid_responses,
        )

    # 5. Return a dictionary that matches the old function's structure
    if extracted_data:
        print(f"[Extracted Assessment Data]:\n{json.dumps(extracted_data, indent=2)}\n")
        return {
            "processed_user_response": extracted_data,
            "next_question_number": question_number + 1,
        }
    else:
        logger.warning("Assessment data extraction pipeline did not produce a result.")
        return {
            "processed_user_response": None,
            "next_question_number": question_number,
        }


async def get_anc_survey_question(user_id: str, user_context: dict) -> dict | None:
    """
    Gets the next contextualized ANC survey question by first determining the
    next logical step, then fetching the content and contextualizing it.
    """
    # TODO: Improve survey histories to know when it's truly a user's first time completing one. For now we are forcing "first_survey" to True.
    user_context["first_survey"] = True
    chat_history = await get_or_create_chat_history(user_id, HistoryType.anc)
    if chat_history:
        current_step = [
            cm.meta["step_title"] for cm in chat_history if cm.role.value == "assistant"
        ][-1]
        next_step = get_next_anc_survey_step(current_step, user_context)
    else:
        next_step = "start"

    is_final = False
    if not next_step:
        is_final = True
        logger.warning(f"End of survey reached! Last step was: {current_step}")
        return None

    text_to_prepend = ""
    if next_step == "not_going_next_one":
        # For this single step, we don't contextualize the content - we just prepend
        # it to the next question.
        question_data = ANC_SURVEY_MAP.get(next_step)
        if not question_data:
            logger.error(f"Could not find question content for step_id: '{next_step}'")
            return None
        text_to_prepend = question_data.content + "\n\n"
        current_step = next_step
        next_step = get_next_anc_survey_step(current_step, user_context)
        if not next_step:
            is_final = True
            logger.warning(f"End of survey reached! Last step was: {current_step}")
            return None

    question_data = ANC_SURVEY_MAP.get(next_step)
    if not question_data:
        logger.error(f"Could not find question content for step_id: '{next_step}'")
        return None

    original_question_content = question_data.content
    valid_responses = question_data.valid_responses
    if not valid_responses:
        if next_step not in [
            "start_going_soon",
            "not_going_next_one",
            "Q_visit_other",
            "Q_challenges_other",
            "Q_why_no_visit_other",
            "Q_why_not_go_other",
            "mom_ANC_remind_me_01",
            "mom_ANC_remind_me_02",
            "end_if_feedback",
            "end",
        ]:
            logger.error(
                f"Could not find valid responses for question for step_id: '{next_step}'"
            )
            return None
        else:
            valid_responses = []

    logger.info(f"Running contextualization pipeline for step: '{next_step}'...")
    contextualized_question = run_anc_survey_contextualization_pipeline(
        user_context,
        chat_history,
        original_question_content,
        valid_responses,
    )

    final_question_text = prepare_valid_responses_to_display_to_anc_survey_user(
        text_to_prepend, contextualized_question, valid_responses, next_step
    )

    return {
        "contextualized_question": final_question_text.strip(),
        "is_final_step": is_final,
        "question_identifier": next_step,
    }


def extract_anc_data_from_response(
    user_response: str,
    user_context: dict,
    step_title: str,
) -> dict:
    """
    Extracts data from a user's response to an ANC survey question.
    """
    logger.info("Running ANC survey data extraction pipeline...")
    question_data = ANC_SURVEY_MAP.get(step_title)
    if not question_data:
        logger.error(f"Could not find question content for step_id: '{step_title}'")
        return user_context
    valid_responses = question_data.valid_responses
    if valid_responses:
        previous_question = prepare_valid_responses_to_use_in_anc_survey_system_prompt(
            question_data.content, valid_responses, step_title
        )
        extracted_data = pipelines.run_clinic_visit_data_extraction_pipeline(
            user_response,
            previous_question,
            valid_responses,
        )
    else:
        extracted_data = user_response
    if extracted_data:
        print(f"[Extracted ANC Data]:\n{json.dumps(extracted_data, indent=2)}\n")
        # Update the user_context with the newly extracted data
        user_context[step_title] = extracted_data
        logger.info(f"Updated user_context for {step_title}: {extracted_data}")
    else:
        logger.warning("ANC data extraction pipeline did not produce a result.")

    return user_context


def detect_user_intent(last_question: str, user_response: str) -> str | None:
    """Runs the intent detection pipeline."""
    logger.info("Running intent detection pipeline...")
    result = pipelines.run_intent_detection_pipeline(last_question, user_response)
    return result.get("intent") if result else None


def get_faq_answer(user_question: str) -> str | None:
    """Runs the FAQ answering pipeline."""
    logger.info("Running FAQ answering pipeline...")
    result = pipelines.run_faq_pipeline(
        user_question,
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


def handle_intro_response(user_input: str, flow_id: str) -> dict:
    """
    Handles a user's response to an introductory consent message by determining
    their intent and validating their answer.
    """
    is_free_text_flow = (
        "onboarding" in flow_id or "behaviour" in flow_id or "survey" in flow_id
    )
    previous_intro_message = (
        doc_store.INTRO_MESSAGES["free_text_intro"]
        if is_free_text_flow
        else doc_store.INTRO_MESSAGES["multiple_choice_intro"]
    )

    # Level 1: General Intent Classification
    intent, intent_related_response = handle_user_message(
        previous_intro_message, user_input
    )

    # --- START DEBUGGING ---
    print("\n--- INTRO DEBUG START ---")
    print(f"DEBUG: User input was '{user_input}'")
    print(f"DEBUG: Intent detected: {intent}")
    # --- END DEBUGGING ---

    action_result = {
        "action": "",
        "message": None,
        "intent": intent,
        "intent_related_response": intent_related_response,
    }

    if intent == "JOURNEY_RESPONSE":
        # Level 2: Validate if the response is "Yes" or "No"
        validated_consent = pipelines.run_clinic_visit_data_extraction_pipeline(
            user_response=user_input,
            previous_service_message=previous_intro_message,
            valid_responses=["Yes", "No"],
        )

        # --- START DEBUGGING ---
        print(f"DEBUG: Validated consent result: {validated_consent}")
        # --- END DEBUGGING ---

        if validated_consent == "Yes":
            action_result["action"] = "PROCEED"
        elif validated_consent == "No":
            action_result["action"] = "ABORT"
            action_result["message"] = doc_store.INTRO_MESSAGES["abort_message"]
        else:
            action_result["action"] = "REPROMPT"
            action_result["message"] = (
                f"Sorry, I didn't quite understand. Please reply with 'Yes' to begin or 'No' to stop.\n\n{previous_intro_message}"
            )

    elif intent in ["QUESTION_ABOUT_STUDY", "HEALTH_QUESTION"]:
        action_result["action"] = "REPROMPT_WITH_ANSWER"
        action_result["message"] = (
            f"{intent_related_response}\n\n{previous_intro_message}"
        )

    else:  # ASK_TO_STOP, CHITCHAT, None, etc. are all treated as declining consent
        action_result["action"] = "ABORT"
        action_result["message"] = doc_store.INTRO_MESSAGES["abort_message"]

    print(f"DEBUG: Final action result: {action_result['action']}")
    print("--- INTRO DEBUG END ---\n")

    return action_result
