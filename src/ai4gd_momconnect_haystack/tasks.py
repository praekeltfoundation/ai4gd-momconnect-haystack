import json
import logging
import re
from typing import Any
from datetime import datetime, timedelta, timezone
from string import ascii_lowercase

from fastapi import HTTPException
from haystack.dataclasses import ChatMessage

from ai4gd_momconnect_haystack.crud import (
    delete_user_journey_state,
    get_assessment_history,
    get_or_create_chat_history,
    get_user_journey_state,
    save_user_journey_state,
)
from ai4gd_momconnect_haystack.enums import AssessmentType, HistoryType, ReminderType
from ai4gd_momconnect_haystack.pipelines import (
    get_next_anc_survey_step,
    run_anc_survey_contextualization_pipeline,
    run_rephrase_question_pipeline,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import (
    AssessmentHistory,
    UserJourneyState,
)

from . import doc_store, pipelines
from .pydantic_models import (
    AssessmentQuestion,
    AssessmentResponse,
    OnboardingResponse,
    ReengagementInfo,
)
from .pydantic_models import (
    LegacySurveyResponse as SurveyResponse,
)
from .utilities import (
    ANC_SURVEY_MAP,
    REMINDER_CONFIG,
    all_onboarding_questions,
    assessment_flow_map,
    assessment_map_to_their_pre,
    create_response_to_key_map,
    kab_b_post_flow_id,
    kab_b_pre_flow_id,
    prepare_valid_responses_to_display_to_anc_survey_user,
    prepare_valid_responses_to_display_to_assessment_user,
    prepare_valid_responses_to_display_to_onboarding_user,
    prepend_valid_responses_with_alphabetical_index,
)

# --- Configuration ---
logger = logging.getLogger(__name__)

# Create a mapping from a field `collects` key to its corresponding question object
FIELD_TO_QUESTION_MAP = {q.collects: q for q in all_onboarding_questions if q.collects}


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
) -> tuple[dict[str, Any], str]:
    """
    Takes user input, extracts data, and returns the fully updated context.
    This is the core business logic for an onboarding turn.
    """
    updated_context = current_context.copy()
    updates = {}
    processed_input = user_input

    current_question_obj = next(
        (
            q
            for q in all_onboarding_questions
            if q.content and q.content in current_question
        ),
        None,
    )

    # Check if this is a simple Yes/No question
    if current_question_obj:
        # Handle Yes/No questions
        if current_question_obj.valid_responses == ["Yes", "No", "Skip"]:
            intent = classify_yes_no_response(user_input)
            if intent == "AFFIRMATIVE":
                updates = {current_question_obj.collects: "Yes"}
            elif intent == "NEGATIVE":
                updates = {current_question_obj.collects: "No"}
        elif current_question_obj.valid_responses:
            # Handle questions with multiple valid responses (e.g., province, area_type)
            user_input_lower = user_input.strip().lower()

            # Rule-based handling for single-letter responses (USSD-style)
            if len(user_input_lower) == 1 and user_input_lower in ascii_lowercase:
                try:
                    response_index = ascii_lowercase.index(user_input_lower)
                    if response_index < len(current_question_obj.valid_responses):
                        selected_response = current_question_obj.valid_responses[
                            response_index
                        ]
                        updates = {current_question_obj.collects: selected_response}
                        processed_input = selected_response
                        logger.info(
                            f"Mapped single-letter '{user_input}' to '{selected_response}'"
                        )
                except (ValueError, IndexError):
                    logger.warning(
                        "Single-letter mapping failed; falling back to keyword/LLM."
                    )

            # If no letter match, attempt keyword extraction
            if not updates:
                for valid_response in current_question_obj.valid_responses:
                    if valid_response.lower() in user_input_lower:
                        updates = {current_question_obj.collects: valid_response}
                        processed_input = valid_response
                        logger.info(
                            f"Keyword match: '{user_input}' -> '{valid_response}'"
                        )
                        break

            # If still no updates, fall back to LLM pipeline
            if not updates:
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

    return updated_context, processed_input


def process_onboarding_step(
    user_input: str, current_context: dict, current_question: str
) -> tuple[dict, dict | None, str]:
    """
    Processes a single step of the onboarding flow for the API.
    """
    updated_context, processed_input = update_context_from_onboarding_response(
        user_input,
        current_context,
        current_question,
    )

    next_question = get_next_onboarding_question(user_context=updated_context)

    return updated_context, next_question, processed_input


def get_assessment_question(
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
    question_history: list[AssessmentHistory] = get_assessment_history(
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


def get_anc_survey_question(
    user_id: str, user_context: dict, chat_history: list[ChatMessage]
) -> dict | None:
    """
    Gets the next contextualized ANC survey question using the reliable,
    persisted journey state.
    """
    # TODO: Improve survey histories to know when it's truly a user's first time completing one. For now we are forcing "first_survey" to True.
    user_context["first_survey"] = True

    # Determine the current step.
    state = get_user_journey_state(user_id)
    current_step = "intro"
    if state and state.current_step_identifier:
        current_step = state.current_step_identifier

    last_question = state.last_question_sent if state else ""

    logger.info(f"get_anc_survey_question: Determined current_step is '{current_step}'")
    next_step = get_next_anc_survey_step(current_step, user_context)

    # --- Consolidated "Going Soon" Logic with 3-Day Reminder ---
    if next_step == "start_going_soon":
        logger.info(
            "Handling 'start_going_soon' step: sending message and scheduling 3-day reminder."
        )
        question_data = ANC_SURVEY_MAP.get("start_going_soon")
        message = question_data.content if question_data else ""

        # Manually set the 3-day reminder delay for this specific path
        time_delta = timedelta(days=3)

        reengagement_info = ReengagementInfo(
            type="SYSTEM_SCHEDULED",
            trigger_at_utc=datetime.now(timezone.utc) + time_delta,
            flow_id="anc-survey",
            reminder_type=ReminderType.SYSTEM_SCHEDULED_THREE_DAY,
        )

        save_user_journey_state(
            user_id=user_id,
            flow_id="anc-survey",
            step_identifier="start_going_soon",
            last_question=message,
            user_context=user_context,
        )

        return {
            "contextualized_question": message,
            "question_identifier": "start_going_soon",
            "is_final_step": True,
            "reengagement_info": reengagement_info,
        }
    elif next_step == "__USER_REQUESTED_REMINDER__":
        logger.info("Handling user-requested reminder.")
        last_step = current_step  # The step the user was on

        current_reminder_count = state.reminder_count if state else 0
        new_reminder_count = current_reminder_count + 1
        reminder_type = ReminderType.USER_REQUESTED
        user_context["reminder_count"] = new_reminder_count

        message, reengagement_info = handle_reminder_request(
            user_id=user_id,
            flow_id="anc-survey",
            step_identifier=last_step,
            last_question=last_question,
            user_context=user_context,
            reminder_type=reminder_type,
        )
        return {
            "contextualized_question": message,
            "question_identifier": next_step,
            "is_final_step": True,
            "reengagement_info": reengagement_info,
        }

    if not next_step:
        logger.warning(f"End of survey reached! Last step was: {current_step}")
        return None

    # --- Check for special action steps before treating it as a question ---
    if next_step.startswith("__") and next_step.endswith("__"):
        logger.info(f"Identified special action step: {next_step}")
        return {
            "contextualized_question": None,  # There is no question to ask the user
            "question_identifier": next_step,  # The action name is returned instead
            "is_final_step": True,  # This action ends the current conversation turn
        }

    is_final = False
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

    is_final = False
    if question_data.content_type in ["end_message", "reminder_confirmation"]:
        is_final = True

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

    # Check if the content type is a reminder; if so, bypass the LLM.
    if question_data.content_type in ["reminder_confirmation", "follow_up_message"]:
        contextualized_question = original_question_content
        logger.info(f"Bypassing contextualization for reminder step: '{next_step}'")
    else:
        # For all other message types, run the contextualization pipeline.
        logger.info(f"Running contextualization pipeline for step: '{next_step}'...")
        contextualized_question = run_anc_survey_contextualization_pipeline(
            user_context,
            chat_history,
            original_question_content,
            valid_responses,
        )

    final_question_text = prepare_valid_responses_to_display_to_anc_survey_user(
        text_to_prepend=text_to_prepend,
        question=contextualized_question,
        valid_responses=valid_responses or [],
        step_title=next_step,
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
    contextualized_question: str,
) -> tuple[dict, dict | None]:
    """
    Extracts data from a user's response to an ANC survey question by mapping
    the response to a standardized key.
    Returns the updated context and an optional action dictionary for the API.
    """
    logger.info("Running confidence-based ANC survey data extraction...")
    question_data = ANC_SURVEY_MAP.get(step_title)
    action_dict = None

    if not question_data:
        logger.error(f"Could not find question content for step_id: '{step_title}'")
        return user_context, None

    if not question_data.valid_responses:
        user_context[step_title] = user_response
        return user_context, None

    # Step 1: Create the mapping from response text to a standardized key
    # FIX: ADD RULE-BASED SHORTCUT FOR SINGLE-LETTER RESPONSES
    response_key_map = create_response_to_key_map(question_data.valid_responses)
    user_response_lower = user_response.strip().lower()

    if len(user_response_lower) == 1 and user_response_lower in ascii_lowercase:
        try:
            response_index = ascii_lowercase.index(user_response_lower)
            if response_index < len(question_data.valid_responses):
                selected_response_text = question_data.valid_responses[response_index]
                standardized_key = response_key_map.get(selected_response_text)

                if standardized_key:
                    logger.info(
                        f"Rule-based match found: '{user_response}' -> '{standardized_key}'"
                    )
                    user_context[step_title] = standardized_key
                    return user_context, None  # Return immediately
        except (ValueError, IndexError):
            logger.warning("Rule-based mapping failed, falling back to LLM.")

    # --- If not handled by rules, proceed with LLM-based extraction ---
    # Step 2: Call the pipeline, passing the key map for the AI to use
    extraction_result = pipelines.run_survey_data_extraction_pipeline(
        user_response=user_response,
        previous_service_message=contextualized_question,
        response_key_map=response_key_map,
    )

    if not extraction_result:
        logger.warning("ANC data extraction pipeline did not produce a result.")
        return user_context, None

    print(f"[Extracted ANC Data]:\n{json.dumps(extraction_result, indent=2)}\n")

    match_type = extraction_result.get("match_type")
    validated_response_key = extraction_result.get("validated_response")
    confidence = extraction_result.get("confidence")

    # Prioritize handling of "no_match" to prevent the loop.
    if match_type == "no_match":
        # Find the standardized key for the "Something else" option.
        other_option_key = next(
            (v for k, v in response_key_map.items() if "Something else" in k), "OTHER"
        )

        user_context[step_title] = other_option_key
        # Save the user's raw, unaltered text in a separate field.
        user_context[f"{step_title}_other_text"] = user_response
        logger.info(
            f"Handled as 'other'. Storing '{user_response}' in '{step_title}_other_text'"
        )

    elif confidence == "low" and isinstance(validated_response_key, str):
        # This block now only runs if it was not a no_match.
        key_to_response_map = {v: k for k, v in response_key_map.items()}
        potential_answer_text = key_to_response_map.get(validated_response_key, "")

        # Only trigger confirmation if there is a valid potential answer to confirm
        if potential_answer_text:
            action_dict = {
                "status": "needs_confirmation",
                "message": f"It sounds like you meant '{potential_answer_text}'. Is that correct?",
                "potential_answer": validated_response_key,
                "step_title_to_confirm": step_title,
            }

    elif validated_response_key:
        # High-confidence, direct match. Store the KEY in the context.
        user_context[step_title] = validated_response_key
        logger.info(f"Updated user_context for {step_title}: {validated_response_key}")

    return user_context, action_dict


def detect_user_intent(
    last_question: str, user_response: str, valid_responses: list[str] | None = None
) -> str | None:
    """Runs the intent detection pipeline."""
    logger.info("Running intent detection pipeline...")
    result = pipelines.run_intent_detection_pipeline(
        last_question, user_response, valid_responses
    )
    return result.get("intent") if result else None


def get_faq_answer(user_question: str) -> str | None:
    """Runs the FAQ answering pipeline."""
    logger.info("Running FAQ answering pipeline...")
    result = pipelines.run_faq_pipeline(
        user_question,
    )
    return result.get("answer") if result else None


def handle_user_message(
    previous_question: str, user_message: str, valid_responses: list[str] | None = None
) -> tuple[str | None, str | None]:
    """
    Orchestrates the process of handling a new user message.
    """
    intent = detect_user_intent(previous_question, user_message, valid_responses)
    response: str | None = None

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
        "SKIP_QUESTION",
    ]:
        pass
    else:
        logger.error(f"Intent detected: {intent}. No specific action defined.")

    return intent, response


def handle_intro_response(user_input: str, flow_id: str) -> dict:
    """
    Handles a user's response to an introductory message, treating negative
    responses as a request to be reminded later.
    """
    is_free_text_flow = (
        "onboarding" in flow_id or "behaviour" in flow_id or "survey" in flow_id
    )
    previous_intro_message = (
        doc_store.INTRO_MESSAGES["free_text_intro"]
        if is_free_text_flow
        else doc_store.INTRO_MESSAGES["multiple_choice_intro"]
    )

    if is_free_text_flow:
        ussd_intent = classify_ussd_intro_response(user_input)
        if ussd_intent == "AFFIRMATIVE":
            return {
                "action": "PROCEED",
                "message": None,
                "intent": "JOURNEY_RESPONSE",
                "intent_related_response": None,
            }
        if ussd_intent == "REMIND_LATER":
            return {
                "action": "PAUSE_AND_REMIND",
                "message": "Great! We’ll remind you tomorrow 🗓️\n\nChat soon 👋🏾",
                "intent": "REQUEST_TO_BE_REMINDED",
                "intent_related_response": None,
            }

    consent_intent = classify_yes_no_response(user_input)
    if consent_intent == "AFFIRMATIVE":
        return {
            "action": "PROCEED",
            "message": None,
            "intent": "JOURNEY_RESPONSE",
            "intent_related_response": None,
        }
    if consent_intent == "NEGATIVE":
        return {
            "action": "PAUSE_AND_REMIND",
            "message": "No problem. We can try again later. Chat soon!",
            "intent": "REQUEST_TO_BE_REMINDED",
            "intent_related_response": None,
        }

    intent, intent_related_response = handle_user_message(
        previous_intro_message, user_input
    )

    action_result = {
        "action": "",
        "message": None,
        "intent": intent,
        "intent_related_response": intent_related_response,
    }

    if intent in ["QUESTION_ABOUT_STUDY", "HEALTH_QUESTION"]:
        action_result["action"] = "REPROMPT_WITH_ANSWER"
        action_result["message"] = (
            f"{intent_related_response}\n\n{previous_intro_message}"
        )
    elif intent == "REQUEST_TO_BE_REMINDED":
        action_result["action"] = "PAUSE_AND_REMIND"
        action_result["message"] = (
            "Of course. I will remind you later. Talk to you soon!"
        )
    else:  # Covers CHITCHAT and other ambiguous cases
        action_result["action"] = "REPROMPT"
        action_result["message"] = (
            f"Sorry, I didn't quite understand. Please reply with 'Yes' to begin or 'No' to stop.\n\n{previous_intro_message}"
        )

    return action_result


def handle_conversational_repair(
    flow_id: str,
    question_identifier: str | int,
    previous_question: str,
    invalid_input: str,
) -> str | None:
    """
    Handles conversational repair. For USSD-style questions, it uses a fixed
    template. For others, it uses an LLM to rephrase.
    """
    # 1. Define which questions are USSD-style (multiple choice)
    is_dma = "dma" in flow_id
    is_kab_mcq = "knowledge" in flow_id or "attitude" in flow_id
    is_anc_mcq = flow_id == "anc-survey" and question_identifier in [
        "start",
        "Q_experience",
        "feedback_if_first_survey",
    ]

    is_ussd_style_question = is_dma or is_kab_mcq or is_anc_mcq

    # 2. Handle the two different repair paths
    if is_ussd_style_question:
        logger.info(
            f"Using template-based repair for USSD-style question: {flow_id} - {question_identifier}"
        )
        # Define the components of the repair message
        ack = "Sorry, we don't understand your answer! Please try again.\n\n"
        instruction = "\n\nPlease reply with the letter corresponding to your answer."

        # Clean any previous repair message components from the incoming question text
        cleaned_question = (
            previous_question.replace(ack, "").replace(instruction, "").strip()
        )

        # Rebuild the message correctly to prevent stacking
        return ack + cleaned_question + instruction
    else:
        # For free-text questions, use the LLM to rephrase
        logger.info(
            f"Using LLM-based repair for free-text question: {flow_id} - {question_identifier}"
        )
        valid_responses: list[str] = []

        # Logic to get valid responses for free-text flows (e.g., Onboarding, KAB-B)
        if flow_id == "onboarding":
            onboarding_question_data = next(
                (
                    q
                    for q in all_onboarding_questions
                    if q.question_number == question_identifier
                ),
                None,
            )
            if onboarding_question_data and onboarding_question_data.valid_responses:
                valid_responses = onboarding_question_data.valid_responses
        elif "behaviour" in flow_id:
            question_list = assessment_flow_map.get(flow_id, [])
            assessment_question_data: AssessmentQuestion | None = next(
                (q for q in question_list if q.question_number == question_identifier),
                None,
            )
            if (
                assessment_question_data
                and assessment_question_data.valid_responses_and_scores
            ):
                valid_responses = [
                    item.response
                    for item in assessment_question_data.valid_responses_and_scores
                ]
        elif flow_id in assessment_flow_map:
            question_list = assessment_flow_map.get(flow_id, [])
            assessment_question_data = next(
                (q for q in question_list if q.question_number == question_identifier),
                None,
            )
            if (
                assessment_question_data
                and assessment_question_data.valid_responses_and_scores
            ):
                valid_responses = [
                    item.response
                    for item in assessment_question_data.valid_responses_and_scores
                ]
        elif flow_id == "anc-survey":
            if isinstance(question_identifier, str):
                question_data = ANC_SURVEY_MAP.get(question_identifier)
                if question_data:
                    valid_responses = question_data.valid_responses or []
                else:
                    logger.error(
                        f"No question data found for ANC step_id: '{question_identifier}'"
                    )
            else:
                logger.error(
                    f"ANC repair called with invalid identifier type: {type(question_identifier)}"
                )

        if valid_responses:
            formatted_responses = prepend_valid_responses_with_alphabetical_index(
                valid_responses
            )
        else:
            formatted_responses = []

        rephrased_question = run_rephrase_question_pipeline(
            previous_question=previous_question,
            invalid_input=invalid_input,
            valid_responses=formatted_responses,
        )

        if not rephrased_question:
            logger.warning("LLM rephrasing failed. Using simple fallback.")
            return f"Sorry, I didn't understand. Please try answering again:\n\n{previous_question}"

        return rephrased_question


def format_user_data_summary_for_whatsapp(user_context: dict) -> str:
    """Formats the collected user data into a human-readable summary for WhatsApp."""
    summary_lines = ["Great, thanks! Here's the information I have for you:"]
    has_data = False

    # Iterate through the defined questions to maintain a consistent order
    for field, question_obj in FIELD_TO_QUESTION_MAP.items():
        value = user_context.get(field)
        # Display the data if it exists and wasn't skipped
        if value and "skip" not in str(value).lower():
            field_title = f"*{field.replace('_', ' ').title()}*"
            summary_lines.append(f"_{field_title}: {value}_")
            has_data = True

    if not has_data:
        return (
            "It looks like we haven't collected any information yet. Let's get started!"
        )

    summary_lines.append("\nIs this all correct?")
    return "\n".join(summary_lines)


def handle_summary_confirmation_step(user_input: str, user_context: dict) -> dict:
    """
    Handles the user's response during the summary confirmation step.
    Signals when onboarding is complete and the next flow should begin.
    """
    updates = pipelines.run_data_update_pipeline(user_input, user_context)

    # If the LLM found specific updates, process them and signal to start DMA.
    if updates:
        user_context.pop("flow_state", None)
        for key, value in updates.items():
            user_context[key] = value
        return {
            "question": "Thank you for the update! Now for the next section.",
            "user_context": user_context,
            "intent": "ONBOARDING_COMPLETE_START_DMA",  # Signal to start DMA
            "results_to_save": list(updates.keys()),
        }

    consent_intent = classify_yes_no_response(user_input)

    # If the user's input is clearly affirmative, complete onboarding and signal to start DMA.
    if consent_intent == "AFFIRMATIVE":
        user_context.pop("flow_state", None)
        return {
            "question": "Perfect, thank you! Now for the next section.",
            "user_context": user_context,
            "intent": "ONBOARDING_COMPLETE_START_DMA",  # Signal to start DMA
            "results_to_save": [],
        }

    # If the user's input is negative or ambiguous, ask for clarification.
    return {
        "question": "No problem. Please tell me what you would like to change. For example, 'My province is Western Cape'.",
        "user_context": user_context,
        "intent": "REPAIR",
        "results_to_save": [],
    }


def classify_yes_no_response(user_input: str) -> str:
    """
    Classifies a user's free-text response as affirmative, negative, or ambiguous
    using robust whole-word matching.

    Returns:
        A string: "AFFIRMATIVE", "NEGATIVE", or "AMBIGUOUS".
    """
    user_input_lower = user_input.lower().strip()
    # Using word boundaries (\b) for more accurate matching
    affirmative_pattern = r"\b(yes|yebo|y|ok|okay|sure|please|definitely|course|yup|yep|yeah|ja|yah|yea|ewe)\b"
    negative_pattern = r"\b(no|nope|n|stop|not|nah|never|nee|hayi)\b"

    is_affirmative = bool(re.search(affirmative_pattern, user_input_lower))
    is_negative = bool(re.search(negative_pattern, user_input_lower))

    # This logic must check for the ambiguous case first.
    if is_affirmative and is_negative:
        return "AMBIGUOUS"
    if is_affirmative:
        return "AFFIRMATIVE"
    if is_negative:
        return "NEGATIVE"

    # If neither is found, the intent is ambiguous.
    return "AMBIGUOUS"


def handle_reminder_request(
    user_id: str,
    flow_id: str,
    step_identifier: str,
    last_question: str,
    user_context: dict,
    reminder_type: int,
) -> tuple[str, ReengagementInfo]:
    """
    Central function to handle pausing a journey. It now uses a dynamic,
    dictionary-based schedule to select the correct reminder.
    """
    # 1. Determine which schedule to use from the central config
    schedule_key = "default"
    if "onboarding" in flow_id:
        schedule_key = "onboarding"
    elif "behaviour" in flow_id:
        schedule_key = "kab"
    elif "survey" in flow_id:
        schedule_key = "survey"

    schedule = REMINDER_CONFIG[schedule_key]

    # 2. Determine which specific reminder config to use from that schedule
    # reminder_count = user_context.get("reminder_count", 0)
    # We'll try and user reminder type instead

    # This logic now explicitly chooses the reminder type based on the flow
    config_key = "DEFAULT"
    if schedule_key in ["onboarding", "kab"] and reminder_type == 2:
        config_key = "FOLLOW_UP"

    reminder_config = schedule[config_key]

    # 3. Create the reminder and save the user's state
    time_delta = reminder_config["delay"]
    reengagement_info = ReengagementInfo(
        type="USER_REQUESTED",
        trigger_at_utc=datetime.now(timezone.utc) + time_delta,
        flow_id=flow_id,
        reminder_type=reminder_type,
    )

    save_user_journey_state(
        user_id=user_id,
        flow_id=flow_id,
        step_identifier=step_identifier,
        last_question=last_question,
        user_context=user_context,
        reminder_type=reminder_type,
    )

    # 4. Return the specific acknowledgement message from the config
    return reminder_config["acknowledgement_message"], reengagement_info


def classify_ussd_intro_response(user_input: str) -> str:
    """
    Classifies a user's response to the USSD-style intro message.
    Returns: "AFFIRMATIVE", "REMIND_LATER", or "AMBIGUOUS".
    """
    user_input_lower = user_input.lower().strip()
    affirmative_pattern = r"\b(a|yes|yebo|start)\b"
    reminder_pattern = r"\b(b|remind|late)\b"

    is_affirmative = bool(re.search(affirmative_pattern, user_input_lower))
    is_reminder = bool(re.search(reminder_pattern, user_input_lower))

    if is_affirmative and not is_reminder:
        return "AFFIRMATIVE"
    if is_reminder and not is_affirmative:
        return "REMIND_LATER"

    return "AMBIGUOUS"


def handle_journey_resumption_prompt(
    user_id: str,
    flow_id: str,
) -> OnboardingResponse | AssessmentResponse | SurveyResponse:
    """
    Handles the logic for a `resume: true` request.

    Determines whether to send a "Ready to continue?" meta-prompt or to
    resume a flow (like ANC survey) directly with the next question.
    """
    state = get_user_journey_state(user_id)
    if not state:
        raise HTTPException(status_code=404, detail="Saved state for user not found.")

    # Determine which schedule to use from the central config
    schedule_key = "default"
    if "onboarding" in flow_id:
        schedule_key = "onboarding"
    elif "behaviour" in flow_id:
        schedule_key = "kab"
    elif "survey" in flow_id:
        schedule_key = "survey"

    schedule = REMINDER_CONFIG.get(schedule_key, REMINDER_CONFIG["default"])
    reminder_count = state.user_context.get("reminder_count", 0)

    config_key = "DEFAULT"
    if (
        schedule_key in ["onboarding", "kab"] and reminder_count >= 1
    ):  # Use FOLLOW_UP for 2nd+ reminder
        config_key = "FOLLOW_UP"

    reminder_config = schedule.get(config_key)
    resume_message = reminder_config.get("resume_message") if reminder_config else None

    # --- Conditional Resumption Logic ---

    if resume_message is None:
        # ANC SURVEY CASE: No resume message, go directly to the next question.
        restored_context = state.user_context
        chat_history = get_or_create_chat_history(user_id, HistoryType.anc)
        question_result = get_anc_survey_question(
            user_id=user_id, user_context=restored_context, chat_history=chat_history
        )

        if not question_result:
            return SurveyResponse(
                question="Welcome back! It looks like you've already completed this survey. Thanks!",
                user_context=restored_context,
                survey_complete=True,
                intent="SYSTEM_RESUMPTION_COMPLETE",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )

        question = question_result.get("contextualized_question", "")
        next_step = question_result.get("question_identifier", "")

        save_user_journey_state(
            user_id=user_id,
            flow_id="anc-survey",
            step_identifier=next_step,
            last_question=state.last_question_sent,
            user_context=restored_context,
        )
        return SurveyResponse(
            question=question,
            user_context=restored_context,
            survey_complete=False,
            intent="SYSTEM_RESUMPTION",
            intent_related_response=None,
            results_to_save=[],
            failure_count=0,
        )
    else:
        # ONBOARDING/KAB CASE: Send the resume meta-prompt.
        save_user_journey_state(
            user_id=user_id,
            flow_id=state.current_flow_id,
            step_identifier="awaiting_reminder_response",
            last_question=state.last_question_sent,
            user_context=state.user_context,
        )

        # Use separate, explicit blocks for each response type
        if "onboarding" in state.current_flow_id:
            return OnboardingResponse(
                question=resume_message,
                user_context=state.user_context,
                intent="SYSTEM_REMINDER_PROMPT",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        elif "survey" in state.current_flow_id:
            return SurveyResponse(
                question=resume_message,
                user_context=state.user_context,
                survey_complete=False,
                intent="SYSTEM_REMINDER_PROMPT",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        else:  # Default to AssessmentResponse for KAB flows
            # Check if the step identifier is a digit before converting to int.
            # If not (e.g., it's 'awaiting_reminder_response'), we can't determine a
            # specific next question number, so we default to None.
            next_q_num = state.next_question_number or 0
            return AssessmentResponse(
                question=resume_message,
                next_question=next_q_num,
                intent="SYSTEM_REMINDER_PROMPT",
                intent_related_response=None,
                processed_answer=None,
                failure_count=0,
            )


def handle_intro_reminder(
    user_id: str,
    flow_id: str,
    user_context: dict,
    last_question: str,
    result: dict,
) -> OnboardingResponse | AssessmentResponse | SurveyResponse:
    """
    Handles the specific case where a user asks for a reminder during an
    introductory message.
    """
    state = get_user_journey_state(user_id)
    current_reminder_count = state.reminder_count if state else 0
    new_reminder_count = current_reminder_count + 1
    reminder_type = ReminderType.USER_REQUESTED
    user_context["reminder_count"] = new_reminder_count

    message, reengagement_info = handle_reminder_request(
        user_id=user_id,
        flow_id=flow_id,
        step_identifier="intro",
        last_question=last_question,
        user_context=user_context,
        reminder_type=reminder_type,
    )

    # Determine which response model to use based on the flow_id
    if "onboarding" in flow_id:
        return OnboardingResponse(
            question=message,
            user_context=user_context,
            intent=result["intent"],
            intent_related_response=None,
            results_to_save=[],
            failure_count=0,
            reengagement_info=reengagement_info,
        )
    elif "survey" in flow_id:
        return SurveyResponse(
            question=message,
            user_context=user_context,
            survey_complete=False,
            intent=result["intent"],
            intent_related_response=None,
            results_to_save=[],
            failure_count=0,
            reengagement_info=reengagement_info,
        )
    else:  # Default to AssessmentResponse
        return AssessmentResponse(
            question=message,
            next_question=0,  # Stay on the intro step
            intent=result["intent"],
            intent_related_response=None,
            processed_answer=None,
            failure_count=0,
            reengagement_info=reengagement_info,
        )


def handle_reminder_response(
    user_id: str,
    user_input: str,
    state: UserJourneyState,  # The user's saved state
) -> SurveyResponse | OnboardingResponse | AssessmentResponse:
    """
    Processes a user's response to a "Ready to continue?" reminder prompt.

    Returns:
        A SurveyResponse object with the next question or another reminder.
    """
    # Use the robust classifier for the user's "Yes" or "Remind me tomorrow"
    intent = classify_ussd_intro_response(user_input)

    if intent == "AFFIRMATIVE":
        # User wants to continue.
        restored_context = state.user_context
        question_to_send = state.last_question_sent
        question_to_send = ""
        next_q_result = None

        # Special case for the 3-day reminder first.
        if state.reminder_type == ReminderType.SYSTEM_SCHEDULED_THREE_DAY:
            question_to_send = state.last_question_sent
        else:
            # The standard behavior is to re-send the last question asked.
            question_to_send = state.last_question_sent

            if not question_to_send:
                chat_history = get_or_create_chat_history(
                    user_id, HistoryType(state.current_flow_id)
                )
                if "survey" in state.current_flow_id:
                    next_q_result = get_anc_survey_question(
                        user_id=user_id,
                        user_context=restored_context,
                        chat_history=chat_history,
                    )
                elif "onboarding" in state.current_flow_id:
                    next_q_result = get_next_onboarding_question(
                        user_context=restored_context
                    )
                else:
                    next_q_result = get_assessment_question(
                        user_id=user_id,
                        flow_id=AssessmentType(state.current_flow_id),
                        question_number=int(state.current_step_identifier) + 1,
                        user_context=restored_context,
                    )
                if next_q_result:
                    question_to_send = next_q_result.get("contextualized_question", "")

        if "onboarding" in state.current_flow_id:
            delete_user_journey_state(user_id)
            return OnboardingResponse(
                question=question_to_send,
                user_context=restored_context,
                intent="JOURNEY_RESUMED",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        elif "survey" in state.current_flow_id:
            save_user_journey_state(
                user_id=user_id,
                flow_id=state.current_flow_id,
                step_identifier=state.expected_step_id,
                last_question=question_to_send,
                user_context=restored_context,
                expected_step_id=state.expected_step_id,
            )
            return SurveyResponse(
                question=question_to_send,
                user_context=restored_context,
                survey_complete=False,
                intent="JOURNEY_RESUMED",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        else:
            # next_q_num = (
            #     (int(state.current_step_identifier) + 1)
            #     if state.current_step_identifier.isdigit()
            #     else int(restored_context.get("next_question_number", 0))
            # )
            delete_user_journey_state(user_id)
            next_q_num = state.next_question_number or 0
            return AssessmentResponse(
                question=question_to_send,
                next_question=next_q_num,
                intent="JOURNEY_RESUMED",
                intent_related_response=None,
                processed_answer=None,
                failure_count=0,
            )

    elif intent == "REMIND_LATER":
        # User wants another reminder. Schedule it.
        # This re-uses the existing reminder logic for DRYness.
        message, reengagement_info = handle_reminder_request(
            user_id=user_id,
            flow_id=state.current_flow_id,
            step_identifier=state.current_step_identifier,
            last_question=state.last_question_sent,
            user_context=state.user_context,
            reminder_type=2,  # This is at least the second reminder
        )
        if "onboarding" in state.current_flow_id:
            return OnboardingResponse(
                question=message,
                user_context=state.user_context,
                intent="REQUEST_TO_BE_REMINDED",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
                reengagement_info=reengagement_info,
            )
        elif "survey" in state.current_flow_id:
            return SurveyResponse(
                question=message,
                user_context=state.user_context,
                survey_complete=False,
                intent="REQUEST_TO_BE_REMINDED",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
                reengagement_info=reengagement_info,
            )
        else:  # Assessments
            return AssessmentResponse(
                question=message,
                next_question=0,
                intent="REQUEST_TO_BE_REMINDED",
                intent_related_response=None,
                processed_answer=None,
                failure_count=0,
                reengagement_info=reengagement_info,
            )

    else:  # Ambiguous response
        # Re-send the reminder prompt to clarify
        schedule_key = "default"
        if "survey" in state.current_flow_id:
            schedule_key = "survey"
        elif "onboarding" in state.current_flow_id:
            schedule_key = "onboarding"
        elif "behaviour" in state.current_flow_id:
            schedule_key = "kab"

        schedule = REMINDER_CONFIG.get(schedule_key, REMINDER_CONFIG["default"])
        rephrased_question = (
            schedule["DEFAULT"].get("resume_message") or "Are you ready to continue?"
        )

        # Return the correct response type for each flow
        if "onboarding" in state.current_flow_id:
            return OnboardingResponse(
                question=rephrased_question,
                user_context=state.user_context,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        elif "survey" in state.current_flow_id:
            return SurveyResponse(
                question=rephrased_question,
                user_context=state.user_context,
                survey_complete=False,
                intent="REPAIR",
                intent_related_response=None,
                results_to_save=[],
                failure_count=0,
            )
        else:  # Assessments
            return AssessmentResponse(
                question=rephrased_question,
                next_question=0,
                intent="REPAIR",
                intent_related_response=None,
                processed_answer=None,
                failure_count=0,
            )


def classify_anc_start_response(user_input: str) -> str | None:
    """
    Reliably classifies the user's response to the first ANC survey question.
    Returns a standardized key: 'YES', 'NO', 'SOON', or None if ambiguous.
    """
    text = user_input.lower().strip()
    print(f"[ANC Start Response]: {text}")

    # Check for alphabetical options first
    if text == "a":
        return "YES"
    if text == "b":
        return "NO"
    if text == "c":
        return "SOON"

    # Check for keywords
    went_pattern = r"\b(yes|went|i went)\b"
    not_going_pattern = r"\b(no|not going)\b"
    going_soon_pattern = r"\b(soon|going soon)\b"

    if re.search(went_pattern, text):
        return "YES"
    if re.search(not_going_pattern, text):
        return "NO"
    if re.search(going_soon_pattern, text):
        print(f"[ANC Start Response]: Detected 'going soon' in response: {text}")
        return "SOON"

    return None  # Return None if the response is ambiguous
