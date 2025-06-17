import logging
from datetime import datetime
from os import environ
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from . import tasks
from .models import AssessmentRun
from .utilities import (
    generate_scenario_id,
    load_json_and_validate,
    save_json_file,
)

load_dotenv()

log_level = environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.

# Define fixed file paths for the Docker environment
DATA_PATH = Path("src/ai4gd_momconnect_haystack/")

DOC_STORE_DMA_PATH = DATA_PATH / "static_content" / "dma.json"
DOC_STORE_KAB_PATH = DATA_PATH / "static_content" / "kab.json"
OUTPUT_PATH = DATA_PATH / "run_output"

# Define which assessments from the doc stores are scorable
SCORABLE_ASSESSMENTS = {
    "dma-assessment",
    "behaviour-assessment",
    "knowledge-assessment",
    "attitude-assessment",
}

# TODO: Align simulation prompts with doc_store for valid responses.
# The prompts used to generate the simulation run (e.g., in contextualization tasks)
# provide response options like "Not at all confident", "A little confident", etc.
# However, the doc_store (e.g., dma.json) expects "I strongly disagree", "I agree", etc.
# for scoring. These must be aligned to ensure correct data extraction and scoring.


def run_simulation():
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    """
    logger.info("--- Starting Haystack POC Simulation ---")
    simulation_results = []

    # --- Simulation ---
    # ** Onboarding Scenario **
    logger.info("\n--- Simulating Onboarding ---")
    user_context = {  # Simulation data collected progressively
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
    chat_history = []
    onboarding_turns = []
    flow_type = "onboarding"

    sim_onboarding = False
    sim_dma = False
    sim_kab = False
    sim_anc_survey = False

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
        # Simulate Onboarding
        for attempt in range(max_onboarding_steps):
            print("-" * 20)
            logger.info(f"Onboarding Question Attempt: {attempt + 1}")

            contextualized_question = tasks.get_next_onboarding_question(
                user_context, chat_history
            )
            if not contextualized_question:
                logger.info("Onboarding flow complete.")
                break

            # Simulate User Response & Data Extraction
            chat_history.append("System to User: " + contextualized_question + "\n> ")
            # Keep getting a user response until it is one that continues the journey:
            intent = None
            while intent != "JOURNEY_RESPONSE":
                chat_history.append(
                    "System to User: " + contextualized_question + "\n> "
                )
                user_response = input(contextualized_question + "\n> ")
                chat_history.append("User to System: " + user_response + "\n> ")

                # Classify user's intent and act accordingly
                intent, message_to_user = tasks.handle_user_message(
                    contextualized_question, user_response
                )
                # If a question about the study or about health was asked, print the
                # response that would be sent to users
                if intent in ["HEALTH_QUESTION", "QUESTION_ABOUT_STUDY"]:
                    print(message_to_user)
                elif intent in [
                    "ASKING_TO_STOP_MESSAGES",
                    "ASKING_TO_DELETE_DATA",
                    "REPORTING_AIRTIME_NOT_RECEIVED",
                ]:
                    print(f"Turn must be notified that user is {intent}")
                elif intent == "CHITCHAT":
                    print(message_to_user)
                    print(
                        (
                            f"User is chitchatting and needs to still respond to "
                            f"the previous question: {contextualized_question}"
                        )
                    )
                else:
                    # intent must be JOURNEY_RESPONSE
                    pass
            chat_history.append("User to System: " + user_response + "\n> ")

            ### Endpoint 2: Turn can call to get extracted data from a user response.
            previous_context = user_context.copy()
            user_context = tasks.extract_onboarding_data_from_response(
                user_response, user_context, chat_history
            )

            # Identify what changed in user_context
            diff_keys = [
                k for k in user_context if user_context[k] != previous_context.get(k)
            ]

            for updated_field in diff_keys:
                # For each piece of extracted info, create a separate turn.
                if updated_field == "other":
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

    # ** DMA Scenario **
    print("")
    while True:
        sim = input("Simulate DMA? (Y/N)\n> ")
        if sim.lower() in ["y", "yes"]:
            sim_dma = True
            break
        elif sim.lower() in ["n", "no"]:
            break
        else:
            print("Please enter 'Y' or 'N'.")

    assessment_turns = []

    if sim_dma:
        logger.info("\n--- Simulating DMA ---")
        user_context["goal"] = "Complete the assessment"
        max_assessment_steps = 10  # Safety break
        question_number = 1

        # Simulate Assessment
        for _ in range(max_assessment_steps):
            print("-" * 20)
            logger.info(f"Assessment Step: Requesting question {question_number}")

            result = tasks.get_assessment_question(
                flow_id="dma-assessment",
                question_number=question_number,
                user_context=user_context,
            )
            if not result:
                logger.info("Assessment flow complete.")
                break
            contextualized_question = result["contextualized_question"]

            # Simulate User Response
            user_response = input(contextualized_question + "\n> ")

            result = tasks.validate_assessment_answer(
                user_response, question_number, "dma-assessment"
            )
            if not result:
                logger.warning(
                    f"Response validation failed for question {question_number}."
                )
                continue
            question_number = result["next_question_number"]

    # ** KAB Scenario **
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
        for flow_id in [
            "knowledge-assessment",
            "attitude-assessment",
            "behaviour-pre-assessment",
        ]:
            logger.info("\n--- Simulating KAB ---")
            question_number = 1
            user_context["goal"] = "Complete the assessment"
            max_assessment_steps = 20  # Safety break

            # Simulate Assessments
            for _ in range(max_assessment_steps):
                print("-" * 20)
                logger.info(f"Assessment Step: Requesting question {question_number}")

                result = tasks.get_assessment_question(
                    flow_id=flow_id,
                    question_number=question_number,
                    user_context=user_context,
                )
                if not result:
                    logger.info("Assessment flow complete.")
                    break
                contextualized_question = result["contextualized_question"]

                # Simulate User Response
                user_response = input(contextualized_question + "\n> ")

                result = tasks.validate_assessment_answer(
                    user_response, question_number, flow_id
                )
                if not result:
                    logger.warning(
                        f"Response validation failed for question {question_number}."
                    )
                    continue
                question_number = result["next_question_number"]

    # ** ANC Survey Scenario **
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
        anc_user_context = {
            "age": user_context.get("age"),
            "gender": user_context.get("gender"),
            "goal": "Complete the ANC survey",
        }
        anc_chat_history = []
        survey_complete = False

        # Simulate ANC Survey
        while not survey_complete:
            print("-" * 20)
            logger.info("ANC Survey Step: Requesting next question...")

            result = tasks.get_anc_survey_question(
                user_context=anc_user_context, chat_history=anc_chat_history
            )

            if not result or not result.get("contextualized_question"):
                logger.info("Could not get next survey question. Ending flow.")
                break

            contextualized_question = result["contextualized_question"]
            survey_complete = result.get("is_final_step", False)

            # Simulate User Response
            anc_chat_history.append(
                "System to User: " + contextualized_question + "\n> "
            )
            user_response = input(contextualized_question + "\n> ")
            anc_chat_history.append("User to System: " + user_response + "\n> ")

            if survey_complete:
                logger.info("Survey flow complete.")
                break

            anc_user_context = tasks.extract_anc_data_from_response(
                user_response, anc_user_context, anc_chat_history
            )
            continue
        # processed_user_response = result['processed_user_response']
        current_assessment_step = result["current_assessment_step"]
        processed_user_response = result["processed_user_response"]
        assessment_turns.append(
            {
                "question_number": current_assessment_step,
                "llm_utterance": contextualized_question,
                "user_utterance": user_response,
                "llm_extracted_user_response": processed_user_response,
            }
        )

    simulation_results.append(
        {
            "scenario_id": generate_scenario_id(
                flow_type=flow_type, username="user_123"
            ),  # TODO: Find a way to pass the username dynamically
            "flow_type": flow_type,
            "turns": assessment_turns,
        }
    )

    logger.info("--- Simulation Complete ---")
    return simulation_results


def _process_run(run: AssessmentRun, all_doc_stores: dict) -> dict:
    """
    Processes a single validated simulation run. It scores the run if applicable,
    otherwise returns the raw data.
    """
    # Guard Clause: If the run is not scorable, return its data immediately.
    if run.flow_type not in SCORABLE_ASSESSMENTS:
        logging.info(f"Appending non-scorable flow to report: {run.flow_type}")
        return run.model_dump()

    logging.info(f"Processing scorable assessment: {run.flow_type}")

    assessment_questions = tasks.load_and_validate_assessment_questions(
        run.flow_type, all_doc_stores
    )

    if not assessment_questions:
        logging.warning(
            f"Could not load/validate questions for '{run.flow_type}'. Appending raw run data."
        )
        return run.model_dump()

    scored_data = tasks.score_assessment_from_simulation(
        [run], run.flow_type, assessment_questions
    )

    if scored_data:
        return {
            "scenario_id": run.scenario_id,
            "flow_type": run.flow_type,
            **scored_data,
        }

    return run.model_dump()


def main() -> None:
    """
    Runs the interactive simulation, orchestrates scoring on the in-memory
    results, and saves the final augmented report.
    """
    logging.info("Starting interactive simulation...")

    # 1. Run the simulation to get the raw output directly in memory.
    raw_simulation_output = run_simulation()
    logging.info("Simulation complete. Starting validation and scoring process...")

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

    all_doc_stores = {**doc_store_dma, **doc_store_kab}

    # 4. Process each validated simulation run using the helper function.
    final_augmented_output = [
        _process_run(run, all_doc_stores) for run in simulation_output
    ]

    # 5. Save the final, augmented report.
    if final_augmented_output:
        # TODO: This section will be replaced with a database write operation.
        # For now, it saves the output to a local JSON file for inspection.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json_file(
            final_augmented_output,
            OUTPUT_PATH / f"final_augmented_output_{timestamp}.json",
        )
        logging.info("Processing complete. Final output generated.")
