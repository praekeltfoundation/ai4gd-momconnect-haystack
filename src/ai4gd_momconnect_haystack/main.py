import json
import logging

from . import tasks


logger = logging.getLogger(__name__)

# TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.


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

    # Simulate Onboarding
    for attempt in range(max_onboarding_steps):
        print("-" * 20)
        logger.info(f"Onboarding Question Attempt: {attempt + 1}")

        ### Endpoint 1: Turn can call to get the next question to send to a user.
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

        ### Endpoint 2: Turn can call to get extracted data from a user response.
        previous_context = user_context.copy()
        user_context = tasks.extract_onboarding_data_from_response(
            user_response, user_context, chat_history
        )

        # Identify what changed in user_context
        diff_keys = [k for k in user_context if user_context[k] != previous_context.get(k)]
        if len(diff_keys) == 1:
            updated_field = diff_keys[0]
            onboarding_turns.append({
                "question_name": updated_field,
                "llm_utterance": contextualized_question,
                "user_utterance": user_response,
                "llm_extracted_user_response": user_context[updated_field]
            })
        else:
            logger.warning(f"Unexpected diff in context at step {attempt + 1}: {diff_keys}")

    simulation_results.append({
        "scenario_id": "onboarding_naledi_v1",  # TODO: Define scenario ID
        "flow_type": "onboarding",
        "turns": onboarding_turns
    })

    # ** Assessment Scenario **
    print("")
    logger.info("\n--- Simulating Assessment ---")
    current_assessment_step = 0
    user_context["goal"] = "Complete the assessment"
    max_assessment_steps = 10  # Safety break
    assessment_turns = []

    # Simulate Assessment
    while current_assessment_step < max_assessment_steps:
        print("-" * 20)
        logger.info(f"Assessment Step: Requesting step after {current_assessment_step}")

        ### Endpoint 3: Turn can call to get an assessment question to send to the user.
        result = tasks.get_assessment_question(
            flow_id="dma_flow_id",
            question_number=current_assessment_step,
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

        ### Endpoint 4: Turn can call to validate a user's response to an assessment question.
        result = tasks.validate_assessment_answer(
            user_response, current_assessment_step
        )
        if not result:
            logger.warning(
                f"Response validation failed for step {current_assessment_step}."
            )
            continue
        # processed_user_response = result['processed_user_response']
        current_assessment_step = result["current_assessment_step"]
        processed_user_response = result["processed_user_response"]
        assessment_turns.append({
            "question_name": current_assessment_step,
            "llm_utterance": contextualized_question,
            "user_utterance": user_response,
            "llm_extracted_user_response": processed_user_response
        })

    simulation_results.append({
        "scenario_id": "assessment_naledi_v1",  # TODO: Define scenario ID
        "flow_type": "dma_assessment",
        "turns": assessment_turns
    })

    logger.info("--- Simulation Complete ---")
    return simulation_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_output = run_simulation()

    # Pretty-print to screen
    pretty_output = json.dumps(run_output, indent=2)
    print("\n--- Simulation Output ---")
    print(pretty_output)
