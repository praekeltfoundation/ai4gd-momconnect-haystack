import asyncio
from datetime import datetime
import logging
import json
from os import environ
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from pydantic import ValidationError
from sqlalchemy import delete

from ai4gd_momconnect_haystack.assessment_logic import (
    _score_single_turn,
    load_and_validate_assessment_questions,
    score_assessment_from_simulation,
    validate_assessment_answer,
    validate_assessment_end_response,
)
from ai4gd_momconnect_haystack.crud import (
    calculate_and_store_assessment_result,
    get_assessment_end_messaging_history,
    get_assessment_result,
    save_assessment_end_message,
    save_assessment_question,
    save_chat_history,
)
from ai4gd_momconnect_haystack.sqlalchemy_models import (
    AssessmentEndMessagingHistory,
    AssessmentHistory,
    AssessmentResultHistory,
    ChatHistory,
)
from ai4gd_momconnect_haystack.tasks import (
    extract_anc_data_from_response,
    extract_onboarding_data_from_response,
    get_anc_survey_question,
    get_assessment_question,
    get_next_onboarding_question,
    handle_user_message,
)
from ai4gd_momconnect_haystack.utilities import (
    assessment_map_to_their_pre,
    assessment_end_flow_map,
    generate_scenario_id,
    load_json_and_validate,
    save_json_file,
)

from .database import AsyncSessionLocal, init_db
from .enums import AssessmentType, HistoryType
from .pydantic_models import (
    AssessmentEndScoreBasedMessage,
    AssessmentEndSimpleMessage,
    AssessmentRun,
    Turn,
)

load_dotenv()

log_level = environ.get("LOGLEVEL", "WARNING").upper()
numeric_level = getattr(logging, log_level, logging.WARNING)

logging.basicConfig(
    level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define fixed file paths for the Docker environment
DATA_PATH = Path("src/ai4gd_momconnect_haystack/")

DOC_STORE_DMA_PATH = DATA_PATH / "static_content" / "dma.json"
DOC_STORE_KAB_PATH = DATA_PATH / "static_content" / "kab.json"
OUTPUT_PATH = DATA_PATH / "run_output"
GT_FILE_PATH = DATA_PATH / "evaluation" / "data" / "ground_truth.json"
SERVICE_PERSONA_PATH = DATA_PATH / "static_content" / "service_persona.json"

SERVICE_PERSONA = load_json_and_validate(SERVICE_PERSONA_PATH, dict)
SERVICE_PERSONA_TEXT = ""
if SERVICE_PERSONA:
    if "persona" in SERVICE_PERSONA.keys():
        SERVICE_PERSONA_TEXT = SERVICE_PERSONA["persona"]

# Define which assessments from the doc stores are scorable
SCORABLE_ASSESSMENTS = {
    "dma-pre-assessment",
    "knowledge-pre-assessment",
    "attitude-pre-assessment",
    "behaviour-pre-assessment",
}


def read_json(filepath: Path) -> dict:
    """Reads JSON data from a file."""
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading JSON from %s: %s", filepath, e)
        raise


async def _get_user_response(
    gt_lookup: dict[str, dict],
    flow_id: str,
    contextualized_question: str,
    turn_identifier_key: str,
    turn_identifier_value: Any,
) -> str | None:
    """
    Gets a user's response, either from ground truth data or from stdin.
    Returns the user's response as a string, or None if a GT turn is not found.
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
            user_response = str(gt_turn.get("user_utterance", ""))
            logger.info(
                f"AUTO-RESPONSE for {turn_identifier_key} "
                f"#{turn_identifier_value}: {user_response}"
            )
            return user_response
        else:
            logger.warning(
                f"Could not find a GT turn for {flow_id} with "
                f"{turn_identifier_key}: {turn_identifier_value}. Ending flow."
            )
            return None
    else:
        return input(contextualized_question + "\n> ")


async def run_simulation(gt_scenarios: list[dict[str, Any]] | None = None):
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    """
    logger.info("--- Starting Haystack POC Simulation ---")
    simulation_results = []

    gt_lookup_by_flow = {}
    if gt_scenarios:
        logger.info("Running in AUTOMATED mode.")
        for scenario in gt_scenarios:
            flow_type = scenario.get("flow_type")
            if flow_type:
                gt_lookup_by_flow[flow_type] = scenario
    else:
        logger.info("Running in INTERACTIVE mode.")

    # --- Simulation ---
    # ** Onboarding Scenario **
    logger.info("\n--- Simulating Onboarding ---")
    user_id = "TestUser"
    user_context: dict[str, Any] = {  # Simulation data collected progressively
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
        chat_history.append(ChatMessage.from_system(text=SERVICE_PERSONA_TEXT))
        # Simulate Onboarding
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

            result = get_next_onboarding_question(user_context, chat_history)

            if not result:
                logger.info("Onboarding flow complete.")
                break

            contextualized_question = result["contextualized_question"]
            question_number = result["question_number"]
            chat_history.append(
                ChatMessage.from_assistant(text=contextualized_question)
            )

            # Simulate User Response & Data Extraction
            # Keep getting a user response until it is one that continues the journey:
            intent = None
            while intent != "JOURNEY_RESPONSE":
                # Get user_response from GT data or input()
                print(f"Question #: {question_number}")
                print(f"Question: {contextualized_question}")
                user_response = await _get_user_response(
                    gt_lookup=gt_lookup_by_flow,
                    flow_id=flow_id,
                    contextualized_question=contextualized_question,
                    turn_identifier_key="question_number",
                    turn_identifier_value=question_number,
                )

                if user_response is None:
                    break

                print(f"User_response: {user_response}")
                chat_history.append(ChatMessage.from_user(text=user_response))

                # Classify user's intent and act accordingly
                intent, intent_related_response = handle_user_message(
                    contextualized_question, user_response
                )
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
                else:
                    # intent must be JOURNEY_RESPONSE
                    pass
            if user_response is None:
                break

            previous_context = user_context.copy()
            user_context = extract_onboarding_data_from_response(
                user_response, user_context, chat_history
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
                                "user_utterance": user_response,
                                "llm_extracted_user_response": value,
                            }
                        )
                else:
                    logger.info(f"Creating turn for extracted field: {updated_field}")
                    onboarding_turns.append(
                        {
                            "question_name": updated_field,
                            "llm_utterance": contextualized_question,
                            "user_utterance": user_response,
                            "llm_extracted_user_response": user_context[updated_field],
                        }
                    )
        run_results["turns"] = onboarding_turns
        simulation_results.append(run_results)

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

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    delete(AssessmentHistory).where(
                        AssessmentHistory.user_id == user_id,
                        AssessmentHistory.assessment_id == flow_id.value,
                    )
                )

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

            result = await get_assessment_question(
                user_id=user_id,
                flow_id=flow_id,
                question_number=question_number,
                user_context=user_context,
            )
            if not result:
                logger.info("Assessment flow complete.")
                break
            contextualized_question = result["contextualized_question"]
            await save_assessment_question(
                user_id=user_id,
                assessment_type=flow_id,
                question_number=question_number,
                user_response=None,
                question=contextualized_question,
                score=None,
            )

            # Simulate User Response
            # Get user_response from GT data or input()
            user_response = ""
            print(f"Question #: {question_number}")
            print(f"Question: {contextualized_question}")
            user_response = await _get_user_response(
                gt_lookup=gt_lookup_by_flow,
                flow_id=flow_id.value,
                contextualized_question=contextualized_question,
                turn_identifier_key="question_number",
                turn_identifier_value=question_number,
            )
            if user_response is None:
                break
            print(f"User_response: {user_response}")

            # Classify user's intent and act accordingly
            intent, intent_related_response = handle_user_message(
                contextualized_question, user_response
            )
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
            else:
                # intent must be JOURNEY_RESPONSE
                pass

            result = validate_assessment_answer(
                user_response, question_number, flow_id.value
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
                    "user_utterance": user_response,
                    "llm_extracted_user_response": processed_user_response,
                }
            )
            question_number = result["next_question_number"]
            assessment_questions = load_and_validate_assessment_questions(
                assessment_map_to_their_pre[flow_id.value]
            )
            if assessment_questions:
                question_lookup = {q.question_number: q for q in assessment_questions}
                score_result = _score_single_turn(
                    Turn.model_validate(dma_turns[-1]),
                    question_lookup,
                )
                await save_assessment_question(
                    user_id=user_id,
                    assessment_type=flow_id,
                    question=None,
                    question_number=dma_turns[-1]["question_number"],
                    user_response=processed_user_response,
                    score=score_result["score"],
                )

        await calculate_and_store_assessment_result(user_id, flow_id)

        # Start of assessment-end messaging
        next_message = "placeholder"
        user_response = ""
        previous_message: str = ""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    delete(AssessmentEndMessagingHistory).where(
                        AssessmentEndMessagingHistory.user_id == user_id,
                        AssessmentEndMessagingHistory.assessment_id == flow_id.value,
                    )
                )
        while next_message:
            assessment_result = await get_assessment_result(user_id, flow_id)
            if not assessment_result:
                logger.error(
                    f"Overall assessment result for {flow_id.value} not found!"
                )
                break
            score_category = assessment_result.category
            if assessment_result.crossed_skip_threshold:
                score_category = "skipped-many"
            messaging_history = await get_assessment_end_messaging_history(
                user_id=user_id, assessment_type=flow_id
            )
            flow_content = assessment_end_flow_map[flow_id.value]
            task = ""
            if not messaging_history:
                previous_message_nr = 0
                # previous_message_data: AssessmentEndItem = None
                previous_message_valid_responses: list[str] | None = []
            else:
                previous_message_nr = messaging_history[-1].message_number
                previous_message_data = [
                    item
                    for item in flow_content
                    if item.message_nr == previous_message_nr
                ][-1]
            # If previous message number was 1, then content is based on score category
            if previous_message_nr == 0:
                pass
            elif previous_message_nr == 1:
                if isinstance(previous_message_data, AssessmentEndScoreBasedMessage):
                    if score_category == "high":
                        previous_message = (
                            previous_message_data.high_score_content.content
                        )
                        previous_message_valid_responses = (
                            previous_message_data.high_score_content.valid_responses
                        )
                    elif score_category == "medium":
                        previous_message = (
                            previous_message_data.medium_score_content.content
                        )
                        previous_message_valid_responses = (
                            previous_message_data.medium_score_content.valid_responses
                        )
                    elif score_category == "low":
                        previous_message = (
                            previous_message_data.low_score_content.content
                        )
                        previous_message_valid_responses = (
                            previous_message_data.low_score_content.valid_responses
                        )
                    else:
                        previous_message = (
                            previous_message_data.skipped_many_content.content
                        )
                        previous_message_valid_responses = (
                            previous_message_data.skipped_many_content.valid_responses
                        )
                else:
                    logger.error("Wrong content type.")
                    break
            # For previous message numbers 2/3, content is the same across score categories.
            else:
                if isinstance(previous_message_data, AssessmentEndSimpleMessage):
                    previous_message = previous_message_data.content
                    previous_message_valid_responses = (
                        previous_message_data.valid_responses
                    )
                else:
                    logger.error("Wrong content type.")
                    break
            if not previous_message_valid_responses and previous_message_nr != 0:
                logger.error("Valid responses not found.")
                break
            if previous_message_nr > 0:
                intent, intent_related_response = handle_user_message(
                    previous_message, user_response
                )
            else:
                intent, intent_related_response = "JOURNEY_RESPONSE", ""
            next_message_nr = previous_message_nr + 1

            # If the user responded to the previous question, we process their response and determine what needs to happen next
            if intent == "JOURNEY_RESPONSE" and user_response:
                # If the user responded to a question that demands a response
                if (
                    (
                        flow_id.value == "dma-pre-assessment"
                        and next_message_nr in [2, 3]
                    )
                    or (
                        flow_id.value == "behaviour-pre-assessment"
                        and next_message_nr == 2
                    )
                    or (
                        flow_id.value == "knowledge-pre-assessment"
                        and next_message_nr == 2
                    )
                    or (
                        flow_id.value == "attitude-pre-assessment"
                        and next_message_nr == 2
                    )
                ):
                    if not previous_message_valid_responses:
                        logger.error("Valid responses not found.")
                        break
                    result = validate_assessment_end_response(
                        previous_message=previous_message,
                        previous_message_nr=next_message_nr - 1,
                        previous_message_valid_responses=previous_message_valid_responses,
                        user_response=user_response,
                    )
                    if result["next_message_number"] == next_message_nr - 1:
                        # validation failed, send previous message again
                        next_message_data = previous_message_data
                    else:
                        processed_user_response = result["processed_user_response"]
                        # If the user response was valid, save it to the existing AssessmentEndMessagingHistory record
                        await save_assessment_end_message(
                            user_response,
                            flow_id,
                            next_message_nr - 1,
                            processed_user_response,
                        )
                        # validation succeeded, so determine next message to send
                        if next_message_nr > len(flow_content):
                            # end of journey reached
                            break
                        next_message_data = [
                            item
                            for item in flow_content
                            if item.message_nr == next_message_nr
                        ][-1]

                        # Now determine the task that's associated with the response
                        if flow_id.value == "dma-pre-assessment":
                            if (
                                previous_message_nr == 1
                                and score_category == "skipped-many"
                            ):
                                if processed_user_response == "Yes":
                                    task = "REMIND_ME_LATER"
                            elif previous_message_nr == 2:
                                task = "STORE_FEEDBACK"
                        elif flow_id.value == "behaviour-pre-assessment":
                            if (
                                previous_message_nr == 1
                                and processed_user_response == "Remind me tomorrow"
                            ):
                                task = "REMIND_ME_LATER"
                        elif flow_id.value == "knowledge-pre-assessment":
                            if (
                                previous_message_nr == 1
                                and processed_user_response == "Remind me tomorrow"
                            ):
                                task = "REMIND_ME_LATER"
                        elif flow_id.value == "attitude-pre-assessment":
                            if previous_message_nr == 1:
                                if score_category == "skipped-many":
                                    if processed_user_response == "Yes":
                                        task = "REMIND_ME_LATER"
                                else:
                                    task = "STORE_FEEDBACK"
                    if isinstance(next_message_data, AssessmentEndSimpleMessage):
                        next_message = next_message_data.content
                    else:
                        logger.error("Wrong content type.")
                        break
                # Else the user responded to the last question, which doesn't require a response
                else:
                    # Here we return an empty message because the user responded by the journey ended
                    next_message = ""

            elif intent == "JOURNEY_RESPONSE":
                # This triggers if the journey is initiating (i.e. the next message to be sent is the first)
                next_message_data = [
                    item for item in flow_content if item.message_nr == next_message_nr
                ][-1]
                if isinstance(next_message_data, AssessmentEndScoreBasedMessage):
                    if score_category == "high":
                        next_message = next_message_data.high_score_content.content
                    elif score_category == "medium":
                        next_message = next_message_data.medium_score_content.content
                    elif score_category == "low":
                        next_message = next_message_data.low_score_content.content
                    else:
                        next_message = next_message_data.skipped_many_content.content
                else:
                    logger.error("Wrong content type.")
                    break
            else:
                # It doesn't seem like the user is responding to the journey, and neither is it their first question, so we ask them the previous question again.
                next_message_data = [
                    item
                    for item in flow_content
                    if isinstance(item, AssessmentEndSimpleMessage)
                    and item.message_nr == previous_message_nr
                ][-1]
                next_message = next_message_data.content

            # If a new question is being sent, save it in a new AssessmentEndMessagingHistory record
            if next_message and next_message_nr == previous_message_nr + 1:
                await save_assessment_end_message(user_id, flow_id, next_message_nr, "")
            print(f"Task: {task}")
            print(f"Question #: {next_message_nr}")
            print(f"Question: {next_message}")
            user_response = await _get_user_response(
                gt_lookup={},
                flow_id=flow_id.value,
                contextualized_question=next_message,
                turn_identifier_key="message_nr",
                turn_identifier_value=next_message_nr,
            )
            if user_response is None:
                break
            print(f"User_response: {user_response}")
        # End of assessment-end messaging

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    delete(AssessmentHistory).where(
                        AssessmentHistory.user_id == user_id,
                        AssessmentHistory.assessment_id == flow_id.value,
                    )
                )
        # And also the overall assessment result, if there is one:
        assessment_result = await get_assessment_result(user_id, flow_id)
        if assessment_result:
            print("Assessment Result:")
            print(assessment_result)
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        delete(AssessmentResultHistory).where(
                            AssessmentResultHistory.user_id == user_id,
                            AssessmentResultHistory.assessment_id == flow_id.value,
                        )
                    )

        run_results["turns"] = dma_turns
        simulation_results.append(run_results)

    # ** KAB Scenario **
    kab_flow_ids = [
        AssessmentType.behaviour_pre_assessment,
        AssessmentType.knowledge_pre_assessment,
        AssessmentType.attitude_pre_assessment,
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
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        delete(AssessmentHistory).where(
                            AssessmentHistory.user_id == user_id,
                            AssessmentHistory.assessment_id == flow_id.value,
                        )
                    )

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
                "turns": None,
            }
            kab_turns = []
            print(f"FlowID: {flow_id}")
            for _ in range(max_assessment_steps):
                print("-" * 20)
                logger.info(f"Assessment Step: Requesting question {question_number}")

                result = await get_assessment_question(
                    user_id=user_id,
                    flow_id=flow_id,
                    question_number=question_number,
                    user_context=user_context,
                )
                if not result:
                    logger.info("Assessment flow complete.")
                    break
                contextualized_question = result["contextualized_question"]
                await save_assessment_question(
                    user_id=user_id,
                    assessment_type=flow_id,
                    question_number=question_number,
                    user_response=None,
                    question=contextualized_question,
                    score=None,
                )

                # Simulate User Response
                # Get user_response from GT data or input()
                print(f"Question #: {question_number}")
                print(f"Question: {contextualized_question}")
                user_response = ""
                user_response = await _get_user_response(
                    gt_lookup=gt_lookup_by_flow,
                    flow_id=flow_id.value,
                    contextualized_question=contextualized_question,
                    turn_identifier_key="question_number",
                    turn_identifier_value=question_number,
                )
                if user_response is None:
                    break
                print(f"User_response: {user_response}")

                # Classify user's intent and act accordingly
                intent, intent_related_response = handle_user_message(
                    contextualized_question, user_response
                )
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
                else:
                    # intent must be JOURNEY_RESPONSE
                    pass

                # Classify user's intent and act accordingly
                intent, intent_related_response = handle_user_message(
                    contextualized_question, user_response
                )
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
                else:
                    # intent must be JOURNEY_RESPONSE
                    pass

                result = validate_assessment_answer(
                    user_response, question_number, flow_id.value
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
                        "user_utterance": user_response,
                        "llm_extracted_user_response": processed_user_response,
                    }
                )
                question_number = result["next_question_number"]
                assessment_questions = load_and_validate_assessment_questions(
                    assessment_map_to_their_pre[flow_id.value]
                )
                if assessment_questions:
                    question_lookup = {
                        q.question_number: q for q in assessment_questions
                    }
                    score_result = _score_single_turn(
                        Turn.model_validate(kab_turns[-1]),
                        question_lookup,
                    )
                    await save_assessment_question(
                        user_id=user_id,
                        assessment_type=flow_id,
                        question=None,
                        question_number=kab_turns[-1]["question_number"],
                        user_response=processed_user_response,
                        score=score_result["score"],
                    )

            await calculate_and_store_assessment_result(user_id, flow_id)

            # Start of assessment-end messaging
            next_message = "placeholder"
            user_response = ""
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        delete(AssessmentEndMessagingHistory).where(
                            AssessmentEndMessagingHistory.user_id == user_id,
                            AssessmentEndMessagingHistory.assessment_id
                            == flow_id.value,
                        )
                    )
            while next_message:
                assessment_result = await get_assessment_result(user_id, flow_id)
                if not assessment_result:
                    logger.error(
                        f"Overall assessment result for {flow_id.value} not found!"
                    )
                    break
                score_category = assessment_result.category
                if assessment_result.crossed_skip_threshold:
                    score_category = "skipped-many"
                messaging_history = await get_assessment_end_messaging_history(
                    user_id=user_id, assessment_type=flow_id
                )
                flow_content = assessment_end_flow_map[flow_id.value]
                task = ""
                if not messaging_history:
                    previous_message_nr = 0
                    # previous_message_data = []
                    previous_message_valid_responses = []
                else:
                    previous_message_nr = messaging_history[-1].message_number
                    previous_message_data = [
                        item
                        for item in flow_content
                        if item.message_nr == previous_message_nr
                    ][-1]
                # If previous message number was 1, then content is based on score category
                if previous_message_nr == 0:
                    pass
                elif previous_message_nr == 1:
                    if isinstance(
                        previous_message_data, AssessmentEndScoreBasedMessage
                    ):
                        if score_category == "high":
                            previous_message = (
                                previous_message_data.high_score_content.content
                            )
                            previous_message_valid_responses = (
                                previous_message_data.high_score_content.valid_responses
                            )
                        elif score_category == "medium":
                            previous_message = (
                                previous_message_data.medium_score_content.content
                            )
                            previous_message_valid_responses = previous_message_data.medium_score_content.valid_responses
                        elif score_category == "low":
                            previous_message = (
                                previous_message_data.low_score_content.content
                            )
                            previous_message_valid_responses = (
                                previous_message_data.low_score_content.valid_responses
                            )
                        else:
                            previous_message = (
                                previous_message_data.skipped_many_content.content
                            )
                            previous_message_valid_responses = previous_message_data.skipped_many_content.valid_responses
                    else:
                        logger.error("Wrong content type.")
                        break
                # For previous message numbers 2/3, content is the same across score categories.
                else:
                    if isinstance(previous_message_data, AssessmentEndSimpleMessage):
                        previous_message = previous_message_data.content
                        previous_message_valid_responses = (
                            previous_message_data.valid_responses
                        )
                    else:
                        logger.error("Wrong content type.")
                        break

                if not previous_message_valid_responses and previous_message_nr != 0:
                    logger.error("Valid responses not found.")
                    break
                if previous_message_nr > 0:
                    intent, intent_related_response = handle_user_message(
                        previous_message, user_response
                    )
                else:
                    intent, intent_related_response = "JOURNEY_RESPONSE", ""
                next_message_nr = previous_message_nr + 1

                # If the user responded to the previous question, we process their response and determine what needs to happen next
                if intent == "JOURNEY_RESPONSE" and user_response:
                    # If the user responded to a question that demands a response
                    if (
                        (
                            flow_id.value == "dma-pre-assessment"
                            and next_message_nr in [2, 3]
                        )
                        or (
                            flow_id.value == "behaviour-pre-assessment"
                            and next_message_nr == 2
                        )
                        or (
                            flow_id.value == "knowledge-pre-assessment"
                            and next_message_nr == 2
                        )
                        or (
                            flow_id.value == "attitude-pre-assessment"
                            and next_message_nr == 2
                        )
                    ):
                        assert previous_message_valid_responses is not None, (
                            "Valid responses not found!"
                        )
                        result = validate_assessment_end_response(
                            previous_message=previous_message,
                            previous_message_nr=next_message_nr - 1,
                            previous_message_valid_responses=previous_message_valid_responses,
                            user_response=user_response,
                        )
                        if result["next_message_number"] == next_message_nr - 1:
                            # validation failed, send previous message again
                            next_message_data = previous_message_data
                        else:
                            processed_user_response = result["processed_user_response"]
                            # If the user response was valid, save it to the existing AssessmentEndMessagingHistory record
                            await save_assessment_end_message(
                                user_response,
                                flow_id,
                                next_message_nr - 1,
                                processed_user_response,
                            )
                            # validation succeeded, so determine next message to send
                            if next_message_nr > len(flow_content):
                                # end of journey reached
                                break
                            next_message_data = [
                                item
                                for item in flow_content
                                if item.message_nr == next_message_nr
                            ][-1]

                            # Now determine the task that's associated with the response
                            if flow_id.value == "dma-pre-assessment":
                                if (
                                    previous_message_nr == 1
                                    and score_category == "skipped-many"
                                ):
                                    if processed_user_response == "Yes":
                                        task = "REMIND_ME_LATER"
                                elif previous_message_nr == 2:
                                    task = "STORE_FEEDBACK"
                            elif flow_id.value == "behaviour-pre-assessment":
                                if (
                                    previous_message_nr == 1
                                    and processed_user_response == "Remind me tomorrow"
                                ):
                                    task = "REMIND_ME_LATER"
                            elif flow_id.value == "knowledge-pre-assessment":
                                if (
                                    previous_message_nr == 1
                                    and processed_user_response == "Remind me tomorrow"
                                ):
                                    task = "REMIND_ME_LATER"
                            elif flow_id.value == "attitude-pre-assessment":
                                if previous_message_nr == 1:
                                    if score_category == "skipped-many":
                                        if processed_user_response == "Yes":
                                            task = "REMIND_ME_LATER"
                                    else:
                                        task = "STORE_FEEDBACK"
                        if isinstance(next_message_data, AssessmentEndSimpleMessage):
                            next_message = next_message_data.content
                        else:
                            logger.error("Wrong content type.")
                            break
                    # Else the user responded to the last question, which doesn't require a response
                    else:
                        # Here we return an empty message because the user responded by the journey ended
                        next_message = ""

                elif intent == "JOURNEY_RESPONSE":
                    # This triggers if the journey is initiating (i.e. the next message to be sent is the first)
                    next_message_data = [
                        item
                        for item in flow_content
                        if item.message_nr == next_message_nr
                    ][-1]
                    if isinstance(next_message_data, AssessmentEndScoreBasedMessage):
                        if score_category == "high":
                            next_message = next_message_data.high_score_content.content
                        elif score_category == "medium":
                            next_message = (
                                next_message_data.medium_score_content.content
                            )
                        elif score_category == "low":
                            next_message = next_message_data.low_score_content.content
                        else:
                            next_message = (
                                next_message_data.skipped_many_content.content
                            )
                    else:
                        logger.error("Wrong content type")
                        break
                else:
                    # It doesn't seem like the user is responding to the journey, and neither is it their first question, so we ask them the previous question again.
                    next_message_data = [
                        item
                        for item in flow_content
                        if isinstance(item, AssessmentEndSimpleMessage)
                        and item.message_nr == previous_message_nr
                    ][-1]
                    next_message = next_message_data.content

                # If a new question is being sent, save it in a new AssessmentEndMessagingHistory record
                if next_message and next_message_nr == previous_message_nr + 1:
                    await save_assessment_end_message(
                        user_id, flow_id, next_message_nr, ""
                    )
                print(f"Task: {task}")
                print(f"Question #: {next_message_nr}")
                print(f"Question: {next_message}")
                user_response = await _get_user_response(
                    gt_lookup={},
                    flow_id=flow_id.value,
                    contextualized_question=next_message,
                    turn_identifier_key="message_nr",
                    turn_identifier_value=next_message_nr,
                )
                if user_response is None:
                    break
                print(f"User_response: {user_response}")
            # End of assessment-end messaging

            async with AsyncSessionLocal() as session:
                async with session.begin():
                    await session.execute(
                        delete(AssessmentHistory).where(
                            AssessmentHistory.user_id == user_id,
                            AssessmentHistory.assessment_id == flow_id.value,
                        )
                    )
            # And also the overall assessment result, if there is one:
            assessment_result = await get_assessment_result(user_id, flow_id)
            if assessment_result:
                async with AsyncSessionLocal() as session:
                    async with session.begin():
                        await session.execute(
                            delete(AssessmentResultHistory).where(
                                AssessmentResultHistory.user_id == user_id,
                                AssessmentResultHistory.assessment_id == flow_id.value,
                            )
                        )

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
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    delete(ChatHistory).where(
                        ChatHistory.user_id == user_id,
                    )
                )
        chat_history.append(ChatMessage.from_system(text=SERVICE_PERSONA_TEXT))
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
            "turns": None,
        }
        anc_survey_turns = []
        for _ in range(max_survey_steps):
            if survey_complete:
                logger.info("Survey flow complete.")
                break

            print("-" * 20)
            logger.info("ANC Survey Step: Requesting next question...")

            result = await get_anc_survey_question(
                user_id=user_id, user_context=anc_user_context
            )

            if not result or not result.get("contextualized_question"):
                logger.info("Could not get next survey question. Ending flow.")
                break

            contextualized_question = result["contextualized_question"]
            question_identifier = result["question_identifier"]
            survey_complete = result.get("is_final_step", False)

            # Simulate User Response
            # Get user_response from GT data or input()
            print(f"Question title: {question_identifier}")
            print(f"Question: {contextualized_question}")
            user_response = ""
            # Find the specific turn in the GT list by searching for the matching question_number
            user_response = await _get_user_response(
                gt_lookup=gt_lookup_by_flow,
                flow_id=flow_id,
                contextualized_question=contextualized_question,
                turn_identifier_key="question_name",
                turn_identifier_value=question_identifier,
            )
            print(f"User_response: {user_response}")
            if user_response is None:
                break

            # Classify user's intent and act accordingly
            intent, intent_related_response = handle_user_message(
                contextualized_question, user_response
            )
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
            else:
                # intent must be JOURNEY_RESPONSE. Perform data extraction to add it and the preceding question to the chat history.
                previous_chat_message = ChatMessage.from_assistant(
                    text=contextualized_question,
                    meta={"step_title": question_identifier},
                )

                previous_context = anc_user_context.copy()
                anc_user_context = extract_anc_data_from_response(
                    user_response,
                    anc_user_context,
                    question_identifier,
                    contextualized_question,
                )
                # Identify what changed in user_context
                diff_keys = [
                    k
                    for k in anc_user_context
                    if anc_user_context[k] != previous_context.get(k)
                ]
                if diff_keys:
                    chat_history.append(previous_chat_message)
                    for updated_field in diff_keys:
                        logger.info(
                            f"Creating turn for extracted field: {updated_field}"
                        )
                        anc_survey_turns.append(
                            {
                                "question_name": question_identifier,
                                "llm_utterance": contextualized_question,
                                "user_utterance": user_response,
                                "llm_extracted_user_response": anc_user_context[
                                    updated_field
                                ],
                            }
                        )
                    # There is only expected to be one updated field:
                    chat_history.append(
                        ChatMessage.from_user(text=anc_user_context[updated_field])
                    )
                    # Save the updated chat_history to the db
                    await save_chat_history(user_id, chat_history, HistoryType.anc)
                else:
                    logger.info(
                        "No valid data extracted... Trying again in the next simulation loop."
                    )
        run_results["turns"] = anc_survey_turns
        simulation_results.append(run_results)

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

    assessment_questions = load_and_validate_assessment_questions(run.flow_type)

    print(f"ASSESSMENT Q: {assessment_questions}")
    if not assessment_questions:
        logging.warning(
            f"Could not load/validate questions for '{run.flow_type}'. Appending raw run data."
        )
        print(
            f"Could not load/validate questions for '{run.flow_type}'. Appending raw run data."
        )
        return run.model_dump()

    scored_data = score_assessment_from_simulation(
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

    # # 1. Run the simulation to get the raw output directly in memory.
    # raw_simulation_output = await run_simulation()
    # logging.info("Simulation complete. Starting validation and scoring process...")
    #
    # Simple check for the '--automated' flag without using argparse
    # is_automated = "--automated"  # Placeholder None
    raw_simulation_output: list[dict[str, Any]] = []

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


# MODIFIED: Allow script to be run directly
if __name__ == "__main__":
    main()
