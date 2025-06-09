import logging

from . import tasks


logger = logging.getLogger(__name__)

# TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.


def run_simulation():
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    """
    logger.info("--- Starting Haystack POC Simulation ---")

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
            user_response = input(contextualized_question + "\n> ")
            chat_history.append("User to System: " + user_response + "\n> ")

            user_context = tasks.extract_onboarding_data_from_response(
                user_response, user_context, chat_history
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

    if sim_dma:
        logger.info("\n--- Simulating DMA ---")
        current_assessment_step = 0
        user_context["goal"] = "Complete the assessment"
        max_assessment_steps = 10  # Safety break

        # Simulate Assessment
        while current_assessment_step < max_assessment_steps:
            print("-" * 20)
            logger.info(
                f"Assessment Step: Requesting step after {current_assessment_step}"
            )

            result = tasks.get_assessment_question(
                flow_id="dma-assessment",
                current_assessment_step=current_assessment_step,
                user_context=user_context,
            )
            if not result:
                logger.info("Assessment flow complete.")
                break
            contextualized_question = result["contextualized_question"]
            current_assessment_step = result["current_question_number"]

            # Simulate User Response
            user_response = input(contextualized_question + "\n> ")

            result = tasks.validate_assessment_answer(
                user_response, current_assessment_step
            )
            if not result:
                logger.warning(
                    f"Response validation failed for step {current_assessment_step}."
                )
                continue
            current_assessment_step = result["current_assessment_step"]

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
            current_assessment_step = 0
            user_context["goal"] = "Complete the assessment"
            max_assessment_steps = 20  # Safety break

            # Simulate Assessments
            while current_assessment_step < max_assessment_steps:
                print("-" * 20)
                logger.info(
                    f"Assessment Step: Requesting step after {current_assessment_step}"
                )

                result = tasks.get_assessment_question(
                    flow_id=flow_id,
                    current_assessment_step=current_assessment_step,
                    user_context=user_context,
                )
                if not result:
                    logger.info("Assessment flow complete.")
                    break
                contextualized_question = result["contextualized_question"]
                current_assessment_step = result["current_question_number"]

                # Simulate User Response
                user_response = input(contextualized_question + "\n> ")

                result = tasks.validate_assessment_answer(
                    user_response, current_assessment_step
                )
                if not result:
                    logger.warning(
                        f"Response validation failed for step {current_assessment_step}."
                    )
                    continue
                current_assessment_step = result["current_assessment_step"]

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

    logger.info("--- Simulation Complete ---")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_simulation()
