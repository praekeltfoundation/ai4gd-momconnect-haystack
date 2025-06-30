import asyncio
import logging
from datetime import datetime
from os import environ
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from pydantic import ValidationError
from sqlalchemy import delete

from ai4gd_momconnect_haystack.sqlalchemy_models import PreAssessmentQuestionHistory

from . import tasks
from .database import AsyncSessionLocal, init_db
from .pydantic_models import AssessmentRun
from .utilities import (
    AssessmentType,
    generate_scenario_id,
    get_pre_assessment_history,
    load_json_and_validate,
    save_json_file,
    read_json,
    save_pre_assessment_question,
)

load_dotenv()

log_level = environ.get("LOGLEVEL", "WARNING").upper()
numeric_level = getattr(logging, log_level, logging.WARNING)

logging.basicConfig(
    level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.

# Define fixed file paths for the Docker environment
DATA_PATH = Path("src/ai4gd_momconnect_haystack/")

DOC_STORE_DMA_PATH = DATA_PATH / "static_content" / "dma.json"
DOC_STORE_KAB_PATH = DATA_PATH / "static_content" / "kab.json"
OUTPUT_PATH = DATA_PATH / "run_output"
GT_FILE_PATH = DATA_PATH / "evaluation" / "data" / "ground_truth.json"
SERVICE_PERSONA_PATH = DATA_PATH / "static_content" / "service_persona.json"

SERVICE_PERSONA = load_json_and_validate(SERVICE_PERSONA_PATH, dict)

# Define which assessments from the doc stores are scorable
SCORABLE_ASSESSMENTS = {
    "dma-pre-assessment",
    "knowledge-pre-assessment",
    "attitude-pre-assessment",
    "behaviour-pre-assessment",
}

# TODO: Align simulation prompts with doc_store for valid responses.
# The prompts used to generate the simulation run (e.g., in contextualization tasks)
# provide response options like "Not at all confident", "A little confident", etc.
# However, the doc_store (e.g., dma.json) expects "I strongly disagree", "I agree", etc.
# for scoring. These must be aligned to ensure correct data extraction and scoring.


async def _get_user_response(
    gt_lookup: dict[str, dict],
    flow_id: str,
    contextualized_question: str,
    turn_identifier_key: str,
    turn_identifier_value: Any,
    use_follow_up: bool = False,
) -> tuple[str | None, dict | None]:
    """
    Gets a user's response, either from ground truth data or from stdin.
    Returns the user's response as a string and the ground truth turn dict.
    """
    if gt_lookup:
        scenario = gt_lookup.get(flow_id, {})
        turns = scenario.get("turns", [])
        gt_turn = next(
            (
                turn
                for turn in turns
                if turn.get(turn_identifier_key) == turn_identifier_value
            ),
            None,
        )
        if gt_turn:
            if use_follow_up:
                user_response = str(gt_turn.get("follow_up_utterance", ""))
            else:
                user_response = str(gt_turn.get("user_utterance", ""))

            if not user_response:
                return None, None

            logger.info(
                f"AUTO-RESPONSE for {turn_identifier_key} "
                f"#{turn_identifier_value} (Follow-up: {use_follow_up}): {user_response}"
            )
            return user_response, gt_turn
        else:
            logger.warning(
                f"Could not find a GT turn for {flow_id} with "
                f"{turn_identifier_key}: {turn_identifier_value}. Ending flow."
            )
            return None, None
    else:
        return input(contextualized_question + "\n> "), None


async def run_simulation(gt_scenarios: list[dict[str, Any]] | None = None):
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    """
    logger.info("--- Starting Haystack POC Simulation ---")
    simulation_results = []

    # ---Pre-process GT data for easy lookup ---
    gt_lookup_by_flow = {}
    if gt_scenarios:
        logger.info("Running in AUTOMATED mode.")
        for scenario in gt_scenarios:
            flow_type = scenario.get("flow_type")
            if flow_type:
                gt_lookup_by_flow[flow_type] = scenario
    else:
        logger.info("Running in INTERACTIVE mode.")

    # ** Onboarding Scenario **
    logger.info("\n--- Simulating Onboarding ---")
    user_id = "TestUser"
    user_context: dict[str, Any] = {
        "age": "33",
        "gender": "female",
        "goal": "Complete the onboarding process",
        "province": None,
        "area_type": None,
        "relationship_status": None,
        "education_level": None,
        "hunger_days": None,
        "num_children": None,
        "phone_ownership": None,
        "other": {},
    }
    max_onboarding_steps = 10  # Safety break
    chat_history: list[ChatMessage] = []
    onboarding_turns = []

    sim_onboarding = "onboarding" in gt_lookup_by_flow
    sim_dma = False
    sim_kab = False
    sim_anc_survey = False

    if not gt_scenarios:
        while True:
            sim = input("Simulate Onboarding? (Y/N)\n> ")
            if sim.lower() in ["y", "yes"]:
                sim_onboarding = True
                break
            elif sim.lower() in ["n", "no"]:
                break
            else:
                print("Please enter 'Y' or 'N'.")

    if sim_onboarding:
        chat_history.append(ChatMessage.from_system(text=SERVICE_PERSONA))
        flow_id = "onboarding"
        gt_scenario = gt_lookup_by_flow.get(flow_id)
        scenario_id_to_use = (
            gt_scenario.get("scenario_id")
            if gt_scenario
            else generate_scenario_id(flow_type=flow_id, username="user_123")
        )
        run_results: dict[str, Any] = {
            "scenario_id": scenario_id_to_use,
            "flow_type": flow_id,
            "turns": None,
        }

        for attempt in range(max_onboarding_steps):
            print("-" * 20)
            logger.info(f"Onboarding Question Attempt: {attempt + 1}")

            result = tasks.get_next_onboarding_question(user_context, chat_history)
            if not result:
                logger.info("Onboarding flow complete.")
                break

            contextualized_question = result["contextualized_question"]
            question_number = result["question_number"]
            chat_history.append(
                ChatMessage.from_assistant(text=contextualized_question)
            )

            print("-" * 20)
            print(f"Question #: {question_number}")

            # Simulate User Response & Data Extraction

            has_deflected = False
            final_user_response = None
            initial_predicted_intent = None
            final_predicted_intent = None
            current_prompt = contextualized_question

            while True:
                user_response, gt_turn = await _get_user_response(
                    gt_lookup=gt_lookup_by_flow,
                    flow_id=flow_id,
                    contextualized_question=current_prompt,
                    turn_identifier_key="question_number",
                    turn_identifier_value=question_number,
                    use_follow_up=has_deflected,
                )

                if user_response is None:
                    break

                print(f"Use_response: {user_response}")
                chat_history.append(ChatMessage.from_user(text=user_response))

                # Classify user's intent and act accordingly
                intent, intent_related_response = tasks.handle_user_message(
                    contextualized_question, user_response
                )
                print(f"Predicted Intent: {intent}")

                if not has_deflected:
                    initial_predicted_intent = intent
                else:
                    final_predicted_intent = intent

                if intent == "JOURNEY_RESPONSE":
                    final_user_response = user_response
                    if not has_deflected:
                        final_predicted_intent = initial_predicted_intent
                    break
                else:
                    if intent in ["HEALTH_QUESTION", "QUESTION_ABOUT_STUDY"]:
                        print(f"Intent: {intent}")
                        print(intent_related_response)
                    elif intent in [
                        "ASKING_TO_STOP_MESSAGES",
                        "ASKING_TO_DELETE_DATA",
                        "REPORTING_AIRTIME_NOT_RECEIVED",
                    ]:
                        print(f"Turn must be notified that user is {intent}")
                    elif intent == "CHITCHAT":
                        print(intent_related_response)
                        print(
                            (
                                f"User is chitchatting and needs to still respond to "
                                f"the previous question: {current_prompt}"
                            )
                        )

                    # Set flag to fetch the follow_up_utterance on the next loop
                    has_deflected = True
                    if not gt_scenario:
                        current_prompt = f"Thanks. To continue, please answer:\n> {contextualized_question}"

            if final_user_response is None:
                continue

            ### Endpoint 2: Turn can call to get extracted data from a user response.
            previous_context = user_context.copy()
            user_context = tasks.extract_onboarding_data_from_response(
                final_user_response, user_context, chat_history
            )

            # Identify what changed in user_context
            diff_keys = [
                k for k in user_context if user_context[k] != previous_context.get(k)
            ]

            for updated_field in diff_keys:
                # For each piece of extracted info, create a separate turn.
                if (
                    updated_field == "other"
                    and isinstance(user_context, dict)
                    and isinstance(previous_context, dict)
                ):
                    # Handle cases where multiple 'other' fields might change
                    other_diff = {
                        k: v
                        for k, v in user_context["other"].items()
                        if v != previous_context["other"].get(k)
                    }
                    for key, value in other_diff.items():
                        logger.info(f"Creating turn for extracted field: {key}")
                        onboarding_turns.append(
                            {
                                "question_name": key,
                                "llm_utterance": contextualized_question,
                                "user_utterance": gt_turn.get("user_utterance")
                                if gt_turn
                                else final_user_response,
                                "follow_up_utterance": gt_turn.get(
                                    "follow_up_utterance"
                                )
                                if gt_turn
                                else None,
                                "llm_initial_predicted_intent": initial_predicted_intent,
                                "llm_final_predicted_intent": final_predicted_intent,
                                "llm_extracted_user_response": value,
                            }
                        )
                else:
                    logger.info(f"Creating turn for extracted field: {updated_field}")
                    onboarding_turns.append(
                        {
                            "question_name": updated_field,
                            "llm_utterance": contextualized_question,
                            "user_utterance": gt_turn.get("user_utterance")
                            if gt_turn
                            else final_user_response,
                            "follow_up_utterance": gt_turn.get("follow_up_utterance")
                            if gt_turn
                            else None,
                            "llm_initial_predicted_intent": initial_predicted_intent,
                            "llm_final_predicted_intent": final_predicted_intent,
                            "llm_extracted_user_response": user_context[updated_field],
                        }
                    )
        run_results["turns"] = onboarding_turns
        simulation_results.append(run_results)

        logger.info("Onboarding Chat History:")
        for msg in chat_history:
            logger.info(f"{msg.role.value}:\n{msg.text}")
        chat_history = []

    # ** DMA Scenario **
    print("")
    sim_dma = "dma-pre-assessment" in gt_lookup_by_flow
    if not gt_scenarios:
        while True:
            sim = input("Simulate DMA? (Y/N)\n> ")
            if sim.lower() in ["y", "yes"]:
                sim_dma = True
                break
            elif sim.lower() in ["n", "no"]:
                break
            else:
                print("Please enter 'Y' or 'N'.")

    if sim_dma:
        logger.info("\n--- Simulating DMA ---")
        flow_id = AssessmentType.dma_pre_assessment
        user_context["goal"] = "Complete the assessment"
        max_assessment_steps = 10  # Safety break
        question_number = 1

        # Simulate Assessment
        gt_scenario = gt_lookup_by_flow.get(flow_id)
        scenario_id_to_use = (
            gt_scenario.get("scenario_id")
            if gt_scenario
            else generate_scenario_id(flow_type=flow_id.value, username="user_123")
        )
        run_results = {
            "scenario_id": scenario_id_to_use,
            "flow_type": flow_id.value,
            "turns": [],
        }
        dma_turns = []

        for _ in range(max_assessment_steps):
            print("-" * 20)
            logger.info(f"Assessment Step: Requesting question {question_number}")

            result = await tasks.get_assessment_question(
                user_id=user_id,
                flow_id=flow_id,
                question_number=question_number,
                user_context=user_context,
            )
            if not result:
                logger.info("Assessment flow complete.")
                break
            contextualized_question = result["contextualized_question"]
            if "-pre" in flow_id.value:
                await save_pre_assessment_question(
                    user_id=user_id,
                    assessment_type=flow_id,
                    question_number=question_number,
                    question=contextualized_question,
                )

            print(f"Question #: {question_number}")
            print(f"Question: {contextualized_question}")

            has_deflected = False
            final_user_response = None
            initial_predicted_intent = None
            final_predicted_intent = None
            current_prompt = contextualized_question

            while True:
                user_response, gt_turn = await _get_user_response(
                    gt_lookup=gt_lookup_by_flow,
                    flow_id=flow_id.value,
                    contextualized_question=contextualized_question,
                    turn_identifier_key="question_number",
                    turn_identifier_value=question_number,
                    use_follow_up=has_deflected,
                )

                if user_response is None:
                    break

                print(f"User_response: {user_response}")
                intent, intent_related_response = tasks.handle_user_message(
                    contextualized_question, user_response
                )
                print(f"Predicted Intent: {intent}")

                if not has_deflected:
                    initial_predicted_intent = intent
                else:
                    final_predicted_intent = intent

                if intent == "JOURNEY_RESPONSE":
                    final_user_response = user_response
                    if not has_deflected:
                        final_predicted_intent = initial_predicted_intent
                    break
                else:
                    # If a question about the study or about health was asked, print the
                    # response that would be sent to users
                    if intent in ["HEALTH_QUESTION", "QUESTION_ABOUT_STUDY"]:
                        print(intent_related_response)
                    elif intent in [
                        "ASKING_TO_STOP_MESSAGES",
                        "ASKING_TO_DELETE_DATA",
                        "REPORTING_AIRTIME_NOT_RECEIVED",
                    ]:
                        print(f"Turn must be notified that user is {intent}")
                    elif intent == "CHITCHAT":
                        print(intent_related_response)
                        print(
                            (
                                f"User is chitchatting and needs to still respond to "
                                f"the previous question: {contextualized_question}"
                            )
                        )

                    # Set flag to fetch the follow_up_utterance on the next loop
                    has_deflected = True
                    if not gt_scenario:
                        current_prompt = f"Thanks. To continue, please answer:\n> {contextualized_question}"

            if final_user_response is None:
                continue

            result = tasks.validate_assessment_answer(
                final_user_response, question_number, flow_id.value
            )
            if not result:
                logger.warning(
                    f"Response validation failed for question {question_number}."
                )
                continue

            processed_user_response = result["processed_user_response"]
            dma_turns.append(
                {
                    "question_number": question_number,
                    "llm_utterance": contextualized_question,
                    "user_utterance": gt_turn.get("user_utterance")
                    if gt_turn
                    else final_user_response,
                    "follow_up_utterance": gt_turn.get("follow_up_utterance")
                    if gt_turn
                    else None,
                    "llm_initial_predicted_intent": initial_predicted_intent,
                    "llm_final_predicted_intent": final_predicted_intent,
                    "llm_extracted_user_response": processed_user_response,
                }
            )
            question_number = result["next_question_number"]

        # Inspect the pre-assessment questions that were stored as history,
        # before deleting them:
        if "-pre" in flow_id.value:
            history = await get_pre_assessment_history(user_id, flow_id)
            if history:
                logger.info(
                    "Stored the following pre-assessment questions during simulation:"
                )
                for q in history:
                    logger.info(f"Question {q.question_number}: {q.question}")
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        delete(PreAssessmentQuestionHistory).where(
                            PreAssessmentQuestionHistory.user_id == user_id,
                            PreAssessmentQuestionHistory.assessment_id == flow_id.value,
                        )
                    )
                    await session.commit()

        run_results["turns"] = dma_turns
        simulation_results.append(run_results)

    # ** KAB Scenario **
    ### Check GT data to decide if KAB should be simulated ###
    kab_flow_ids = [
        AssessmentType.knowledge_pre_assessment,
        AssessmentType.attitude_pre_assessment,
        AssessmentType.behaviour_pre_assessment,
    ]
    kab_flows_in_gt = [flow for flow in kab_flow_ids if flow.value in gt_lookup_by_flow]
    sim_kab = bool(kab_flows_in_gt)
    if not gt_scenarios:
        print("")
        while True:
            sim = input("Simulate KAB? (Y/N)\n> ")
            if sim.lower() in ["y", "yes"]:
                sim_kab = True
                break
            elif sim.lower() in ["n", "no"]:
                break
            else:
                print("Please enter 'Y' or 'N'.")

    if sim_kab:
        flows_to_run = kab_flows_in_gt if gt_scenarios else kab_flow_ids

        for flow_id in flows_to_run:
            logger.info("\n--- Simulating KAB ---")
            question_number = 1
            user_context["goal"] = "Complete the assessment"
            max_assessment_steps = 20  # Safety break

            # Simulate Assessments
            gt_scenario = gt_lookup_by_flow.get(flow_id)
            scenario_id_to_use = (
                gt_scenario.get("scenario_id")
                if gt_scenario
                else generate_scenario_id(flow_type=flow_id.value, username="user_123")
            )
            run_results = {
                "scenario_id": scenario_id_to_use,
                "flow_type": flow_id.value,
                "turns": [],
            }
            kab_turns = []
            print(f"FlowID: {flow_id}")

            for _ in range(max_assessment_steps):
                print("-" * 20)
                logger.info(f"Assessment Step: Requesting question {question_number}")

                result = await tasks.get_assessment_question(
                    user_id=user_id,
                    flow_id=flow_id,
                    question_number=question_number,
                    user_context=user_context,
                )
                if not result:
                    logger.info("Assessment flow complete.")
                    break
                contextualized_question = result["contextualized_question"]
                if "-pre" in flow_id.value:
                    await save_pre_assessment_question(
                        user_id=user_id,
                        assessment_type=flow_id,
                        question_number=question_number,
                        question=contextualized_question,
                    )

                print("-" * 20)
                print(f"Question #: {question_number}")
                print(f"Question: {contextualized_question}")

                has_deflected = False
                final_user_response = None
                initial_predicted_intent = None
                final_predicted_intent = None
                current_prompt = contextualized_question

                while True:
                    user_response, gt_turn = await _get_user_response(
                        gt_lookup=gt_lookup_by_flow,
                        flow_id=flow_id.value,
                        contextualized_question=contextualized_question,
                        turn_identifier_key="question_number",
                        turn_identifier_value=question_number,
                        use_follow_up=has_deflected,
                    )
                    if user_response is None:
                        break
                    print(f"Use_response: {user_response}")
                    intent, intent_related_response = tasks.handle_user_message(
                        contextualized_question, user_response
                    )
                    print(f"Predicted Intent: {intent}")

                    if not has_deflected:
                        initial_predicted_intent = intent
                    else:
                        final_predicted_intent = intent

                    if intent == "JOURNEY_RESPONSE":
                        final_user_response = user_response
                        if not has_deflected:
                            final_predicted_intent = initial_predicted_intent
                        break
                    else:
                        if intent in ["HEALTH_QUESTION", "QUESTION_ABOUT_STUDY"]:
                            print(intent_related_response)
                        elif intent in [
                            "ASKING_TO_STOP_MESSAGES",
                            "ASKING_TO_DELETE_DATA",
                            "REPORTING_AIRTIME_NOT_RECEIVED",
                        ]:
                            print(f"Turn must be notified that user is {intent}")
                        elif intent == "CHITCHAT":
                            print(intent_related_response)
                            print(
                                (
                                    f"User is chitchatting and needs to still respond to "
                                    f"the previous question: {contextualized_question}"
                                )
                            )

                        has_deflected = True
                        if not gt_scenario:
                            current_prompt = f"Thanks. To continue, please answer:\n> {contextualized_question}"

                if final_user_response is None:
                    continue

                result = tasks.validate_assessment_answer(
                    final_user_response, question_number, flow_id.value
                )
                if not result:
                    logger.warning(
                        f"Response validation failed for question {question_number}."
                    )
                    continue
                processed_user_response = result["processed_user_response"]
                kab_turns.append(
                    {
                        "question_number": question_number,
                        "llm_utterance": contextualized_question,
                        "user_utterance": gt_turn.get("user_utterance")
                        if gt_turn
                        else final_user_response,
                        "follow_up_utterance": gt_turn.get("follow_up_utterance")
                        if gt_turn
                        else None,
                        "llm_initial_predicted_intent": initial_predicted_intent,
                        "llm_final_predicted_intent": final_predicted_intent,
                        "llm_extracted_user_response": processed_user_response,
                    }
                )
                question_number = result["next_question_number"]

            if "-pre" in flow_id.value:
                history = await get_pre_assessment_history(user_id, flow_id)
                if history:
                    logger.info(
                        "Stored the following pre-assessment questions during simulation:"
                    )
                    for q in history:
                        logger.info(f"Question {q.question_number}: {q.question}")
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        await session.execute(
                            delete(PreAssessmentQuestionHistory).where(
                                PreAssessmentQuestionHistory.user_id == user_id,
                                PreAssessmentQuestionHistory.assessment_id
                                == flow_id.value,
                            )
                        )
                        await session.commit()

            run_results["turns"] = kab_turns
            simulation_results.append(run_results)

    # ** ANC Survey Scenario **
    sim_anc_survey = "anc-survey" in gt_lookup_by_flow
    if not gt_scenarios:
        print("")
        while True:
            sim = input("Simulate ANC Survey? (Y/N)\n> ")
            if sim.lower() in ["y", "yes"]:
                sim_anc_survey = True
                break
            elif sim.lower() in ["n", "no"]:
                break
            else:
                print("Please enter 'Y' or 'N'.")

    if sim_anc_survey:
        logger.info("\n--- Simulating ANC Survey ---")
        chat_history.append(ChatMessage.from_system(text=SERVICE_PERSONA))
        anc_user_context = {
            "age": user_context.get("age"),
            "gender": user_context.get("gender"),
            "goal": "Complete the ANC survey",
        }
        survey_complete = False
        max_survey_steps = 25  # Safety break

        # Simulate ANC Survey
        flow_id = "anc-survey"
        gt_scenario = gt_lookup_by_flow.get(flow_id)
        scenario_id_to_use = (
            gt_scenario.get("scenario_id")
            if gt_scenario
            else generate_scenario_id(flow_type=flow_id, username="user_123")
        )
        run_results = {
            "scenario_id": scenario_id_to_use,
            "flow_type": flow_id,
            "turns": [],
        }
        anc_survey_turns = []
        for _ in range(max_survey_steps):
            if survey_complete:
                logger.info("Survey flow complete.")
                break

            print("-" * 20)
            logger.info("ANC Survey Step: Requesting next question...")

            result = tasks.get_anc_survey_question(
                user_context=anc_user_context, chat_history=chat_history
            )

            if not result or not result.get("contextualized_question"):
                logger.info("Could not get next survey question. Ending flow.")
                break

            contextualized_question = result["contextualized_question"]
            question_identifier = result["question_identifier"]
            survey_complete = result.get("is_final_step", False)
            chat_history.append(
                ChatMessage.from_assistant(text=contextualized_question)
            )

            print("-" * 20)
            print(f"Question title: {question_identifier}")
            print(f"Question: {contextualized_question}")

            has_deflected = False
            final_user_response = None
            initial_predicted_intent = None
            final_predicted_intent = None
            current_prompt = contextualized_question

            while True:
                user_response, gt_turn = await _get_user_response(
                    gt_lookup=gt_lookup_by_flow,
                    flow_id=flow_id,
                    contextualized_question=contextualized_question,
                    turn_identifier_key="question_name",
                    turn_identifier_value=question_identifier,
                    use_follow_up=has_deflected,
                )
                if user_response is None:
                    break

                print(f"Use_response: {user_response}")
                chat_history.append(ChatMessage.from_user(text=user_response))
                intent, intent_related_response = tasks.handle_user_message(
                    contextualized_question, user_response
                )
                print(f"Predicted Intent: {intent}")

                if not has_deflected:
                    initial_predicted_intent = intent
                else:
                    final_predicted_intent = intent

                if intent == "JOURNEY_RESPONSE":
                    final_user_response = user_response
                    if not has_deflected:
                        final_predicted_intent = initial_predicted_intent
                    break
                else:
                    if intent in ["HEALTH_QUESTION", "QUESTION_ABOUT_STUDY"]:
                        print(intent_related_response)
                    elif intent in [
                        "ASKING_TO_STOP_MESSAGES",
                        "ASKING_TO_DELETE_DATA",
                        "REPORTING_AIRTIME_NOT_RECEIVED",
                    ]:
                        print(f"Turn must be notified that user is {intent}")
                    elif intent == "CHITCHAT":
                        print(intent_related_response)
                        print(
                            (
                                f"User is chitchatting and needs to still respond to "
                                f"the previous question: {contextualized_question}"
                            )
                        )

                    has_deflected = True
                    if not gt_scenario:
                        current_prompt = f"Thanks. To continue, please answer:\n> {contextualized_question}"

            if final_user_response is None:
                continue

            previous_context = anc_user_context.copy()
            anc_user_context = tasks.extract_anc_data_from_response(
                final_user_response, anc_user_context, chat_history
            )
            # Identify what changed in user_context
            diff_keys = [
                k
                for k in anc_user_context
                if anc_user_context[k] != previous_context.get(k)
            ]
            for updated_field in diff_keys:
                logger.info(f"Creating turn for extracted field: {updated_field}")
                anc_survey_turns.append(
                    {
                        "question_name": question_identifier,
                        "llm_utterance": contextualized_question,
                        "user_utterance": gt_turn.get("user_utterance")
                        if gt_turn
                        else final_user_response,
                        "follow_up_utterance": gt_turn.get("follow_up_utterance")
                        if gt_turn
                        else None,
                        "llm_initial_predicted_intent": initial_predicted_intent,
                        "llm_final_predicted_intent": final_predicted_intent,
                        "llm_extracted_user_response": anc_user_context[updated_field],
                    }
                )
        run_results["turns"] = anc_survey_turns
        simulation_results.append(run_results)

        logger.info("ANC Survey Chat History:")
        for msg in chat_history:
            logger.info(f"{msg.role.value}:\n{msg.text}")
        chat_history = []

    # --- End of Simulation ---

    logger.info("--- Simulation Complete ---")
    return simulation_results


def _process_run(run: AssessmentRun) -> dict:
    """
    Processes a single validated simulation run. It scores the run if applicable,
    otherwise returns the raw data.
    """
    # Guard Clause: If the run is not scorable, return its data immediately.
    if run.flow_type not in SCORABLE_ASSESSMENTS:
        logging.info(f"Appending non-scorable flow to report: {run.flow_type}")
        return run.model_dump()

    logging.info(f"Processing scorable assessment: {run.flow_type}")

    assessment_questions = tasks.load_and_validate_assessment_questions(run.flow_type)

    print(f"ASSESSMENT Q: {assessment_questions}")
    if not assessment_questions:
        logging.warning(
            f"Could not load/validate questions for '{run.flow_type}'. Appending raw run data."
        )
        print(
            f"Could not load/validate questions for '{run.flow_type}'. Appending raw run data."
        )
        return run.model_dump()

    scored_data = tasks.score_assessment_from_simulation(
        [run], run.flow_type, assessment_questions
    )
    print(f"ASSESSMENT Q: {assessment_questions}")

    if scored_data:
        return {
            "scenario_id": run.scenario_id,
            "flow_type": run.flow_type,
            **scored_data,
        }

    return run.model_dump()


async def async_main(
    RESULT_PATH=OUTPUT_PATH,
    GT_FILE_PATH=GT_FILE_PATH,
    is_automated=False,
    save_simulation=False,
) -> None:
    """
    Runs the interactive simulation, orchestrates scoring on the in-memory
    results, and saves the final augmented report.
    """
    logging.info("Initializing database...")
    await init_db()
    logging.info("Database initialized.")
    logging.info("Starting interactive simulation...")

    # 1. Run the simulation to get the raw output directly in memory.
    if is_automated:
        logging.info("Starting simulation in AUTOMATED mode...")
        gt_data = read_json(GT_FILE_PATH)
        gt_scenarios_from_json: list | None = gt_data.get("scenarios")

        gt_scenarios = []
        if gt_scenarios_from_json:
            gt_scenarios = [s for s in gt_scenarios_from_json if s.get("enabled", True)]
            # l_ = [s.get("selected_for_evaluation", True) for s in gt_scenarios_from_json]

        if not gt_scenarios:
            logging.critical(
                f"Failed to load ground truth file from {GT_FILE_PATH}. Exiting."
            )
            return
        raw_simulation_output = await run_simulation(gt_scenarios=gt_scenarios)
    else:
        logging.info("Starting simulation in INTERACTIVE mode...")
        raw_simulation_output = await run_simulation(gt_scenarios=None)

    print("_________ RUN OUPUT__________")
    print(raw_simulation_output)

    # 2. Validate the IN-MEMORY raw output directly using Pydantic.
    try:
        simulation_output = [
            AssessmentRun.model_validate(run) for run in raw_simulation_output
        ]
        logging.info("Simulation output validated successfully.")
    except ValidationError as e:
        logging.critical(f"Simulation produced invalid output data:\n{e}")
        return

    # 3. Load the doc stores which contain the assessment questions and rules.
    doc_store_dma = load_json_and_validate(DOC_STORE_DMA_PATH, dict)
    doc_store_kab = load_json_and_validate(DOC_STORE_KAB_PATH, dict)

    if doc_store_dma is None or doc_store_kab is None:
        logging.critical(
            "Could not load one or more required doc store files. Exiting."
        )
        return

    # 4. Process each validated simulation run using the helper function.
    final_augmented_output = [_process_run(run) for run in simulation_output]

    # 5. Save the final, augmented report.
    if save_simulation and final_augmented_output:
        # TODO: This section will be replaced with a database write operation.
        # For now, it saves the output to a local JSON file for inspection.
        file_extention = datetime.now().strftime("%y%m%d-%H%M")
        SIMULATION_FILE_PATH = (
            RESULT_PATH / f"simulation_run_results_{file_extention}.json"
        )

        save_json_file(
            final_augmented_output,
            SIMULATION_FILE_PATH,
        )
        logging.info(
            f"Processing complete. Final output generated and save to {SIMULATION_FILE_PATH}"
        )
        return SIMULATION_FILE_PATH


def main() -> None:
    """The synchronous entry point for the script."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nSimulation cancelled by user.")


if __name__ == "__main__":
    main()
