import json
import logging
from datetime import datetime
from typing import Any

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
    flow_type = "onboarding"

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
        diff_keys = [
            k for k in user_context
            if user_context[k] != previous_context.get(k)
        ]
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
        "scenario_id": generate_scenario_id(
            flow_type=flow_type, username="user_123"
        ),   # TODO: Find a way to pass the username dynamically
        "flow_type": flow_type,
        "turns": onboarding_turns
    })

    # ** Assessment Scenario **
    print("")
    logger.info("\n--- Simulating Assessment ---")
    current_assessment_step = 0
    user_context["goal"] = "Complete the assessment"
    max_assessment_steps = 10  # Safety break
    assessment_turns = []
    flow_type = "dma_assessment"

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
        "scenario_id": generate_scenario_id(
            flow_type=flow_type, username="user_123"
        ),   # TODO: Find a way to pass the username dynamically
        "flow_type": flow_type,
        "turns": assessment_turns
    })

    logger.info("--- Simulation Complete ---")
    return simulation_results


def generate_scenario_id(
    flow_type: str, username: str
) -> str:
    """
    Generates a unique scenario ID based on flow type, name, version,
    and current timestamp.
    """
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return f"{flow_type}_{username}_{timestamp}"


def _score_single_run(
    run_result: dict[str, Any],
    scorable_assessments_map: dict[str, list],
    full_simulation_output: list[dict[str, Any]]
) -> None:
    """
    Scores a single run from the simulation if it's a scorable assessment.
    This function modifies the run_result dictionary in-place.

    Args:
        run_result: A single dictionary from the list of simulation outputs.
        scorable_assessments_map: A mapping of assessment IDs to their questions.
        full_simulation_output: The complete list of all simulation runs.
    """
    flow_type = run_result.get("flow_type")

    # Exit early if the flow_type isn't in our map of scorable assessments
    if flow_type not in scorable_assessments_map:
        return

    logging.info(f"Found scorable assessment: '{flow_type}'. Scoring now...")
    assessment_questions = scorable_assessments_map[flow_type]

    try:
        # Pass the full simulation output so the scoring function can
        # find the run it needs internally.
        scoring_summary = tasks.score_assessment_from_simulation(
            simulation_output=full_simulation_output,
            assessment_id=flow_type,
            assessment_questions=assessment_questions,
        )

        if scoring_summary:
            # Merge the summary scores (e.g., 'user_total_score')
            # back into the original run dictionary.
            run_result.update(scoring_summary)
            logging.info(f"Successfully scored and updated '{flow_type}'.")

    except Exception as e:
        # Robustly catch any unexpected errors during the scoring process
        logging.error(f"An unexpected error occurred while scoring '{flow_type}': {e}", exc_info=True)
        run_result['scoring_error'] = str(e)



def main():
    """
    Runs the simulation, scores all applicable assessments, and logs the
    final, augmented output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 1. Define all scorable assessments and their corresponding questions.
    # This mapping makes the process scalable and easy to maintain.
    scorable_assessments_map = {
        "dma_assessment": tasks.all_dma_questions,
        # "behaviour-assessment": tasks.all_behaviour_questions, # Uncomment if needed
        # "knowledge-assessment": tasks.all_knowledge_questions,
        # "attitude-assessment": tasks.all_attitude_questions,
    }

    # 2. Run the simulation to get the raw output.
    logging.info("Starting simulation...")
    run_output = run_simulation()
    logging.info("Simulation complete.")

    # 3. Score each result in the simulation output.
    logging.info("Scoring all applicable assessments...")
    for run in run_output:
        _score_single_run(run, scorable_assessments_map, run_output)

    # 4. Log the final, augmented output.
    print("\n--- Final Augmented Simulation Output ---")
    final_output_json = json.dumps(run_output, indent=2)
    print(final_output_json)

    logging.info("Processing complete. Final output generated.")
