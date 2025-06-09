# Save this as batch_evaluator_final.py
import logging
import json
from pathlib import Path
from langfuse import Langfuse

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_golden_dataset(path: Path) -> list:
    """Loads the golden dataset from a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load golden dataset from {path}: {e}")
        return []


def run_batch_evaluation(scenarios_to_evaluate: list) -> None:
    """
    Performs a final, robust turn-by-turn evaluation.
    This script is updated to align with the final ground truth schema.
    """
    logging.info("--- Starting Final Batch Evaluation Script ---")
    langfuse = Langfuse()

    if not scenarios_to_evaluate:
        return

    for scenario in scenarios_to_evaluate:
        session_id_from_file = scenario.get("scenario_id")
        flow_type = scenario.get("flow_type")
        ground_truth_turns = scenario.get("turns", [])
        if not session_id_from_file or not flow_type:
            continue

        # _____ StopIteration
        latest_session_id = None
        if not session_id_from_file or not flow_type:
            continue

        session_id_to_evaluate = None

        # --- NEW: Logic to handle the "LATEST" keyword ---
        if session_id_from_file == "LATEST":
            if latest_session_id is None:  # Fetch only if we haven't already
                logger.info("'LATEST' keyword found. Fetching most recent session...")
                try:
                    latest_sessions = langfuse.get_sessions(limit=1)
                    if latest_sessions:
                        latest_session_id = latest_sessions[0].id
                        logger.info(f"    ...found latest session: {latest_session_id}")
                    else:
                        logger.warning(
                            "Could not find any sessions in the project to evaluate."
                        )
                        continue  # Skip this scenario
                except Exception as e:
                    logger.error(
                        f"An error occurred while fetching the latest session: {e}"
                    )
                    continue  # Skip this scenario
            session_id_to_evaluate = latest_session_id
        else:
            # If it's not "LATEST", use the ID from the file as usual
            session_id_to_evaluate = session_id_from_file

        if not session_id_to_evaluate:
            continue

        logger.info(
            f"\n>>> Evaluating Session: {session_id_to_evaluate} (Flow: {flow_type}) <<<"
        )

        try:
            session = langfuse.get_session(session_id_to_evaluate)
            traces_map = {trace.name: trace for trace in session.traces}
        except Exception:
            logger.warning(
                f"Session '{session_id_to_evaluate}' not found in Langfuse. Skipping."
            )
            continue

        for turn in ground_truth_turns:
            flow_type = turn.get("flow_type")
            q_num = turn.get("question_number")
            user_utterance = turn.get("user_utterance")
            ground_truth_delta = turn.get("ground_truth_delta")

            # Add a safety check
            if not flow_type or q_num is None or ground_truth_delta is None:
                continue

            # Construct the expected trace name using the turn's specific flow_type
            expected_trace_name = f"{flow_type}-q{q_num}"
            actual_trace = traces_map.get(expected_trace_name)

            if not actual_trace:
                logger.warning(
                    f"    Trace '{expected_trace_name}' not found. Skipping."
                )
                continue

            logger.info(
                f"  - Scoring trace: '{actual_trace.name}' (ID: {actual_trace.id})"
            )

            actual_output = actual_trace.output
            is_match = actual_output == ground_truth_delta
            score_value = 1 if is_match else 0
            comment = (
                "SUCCESS: Match."
                if is_match
                else f"FAILURE on user utterance: '{user_utterance}'. Expected {ground_truth_delta}, but got {actual_output}"
            )

            score_name = f"{flow_type}_accuracy"

            langfuse.score(
                trace_id=actual_trace.id,
                name=score_name,
                value=score_value,
                comment=comment,
            )
            logger.info(f"    Applied score '{score_name}' with value {score_value}.")

    logging.info("\n--- Batch Evaluation Finished ---")
    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    # Point this to the actual path of your final golden dataset
    SCRIPT_DIRECTORY = Path(__file__).resolve().parent
    DATASET_PATH = SCRIPT_DIRECTORY / "evaluation" / "data" / "golden_dataset.json"
    if not DATASET_PATH.exists():
        logger.error(
            f"CRITICAL: Golden dataset not found at expected path: {DATASET_PATH}"
        )
        logger.error(
            "Please ensure the 'evaluation' directory is in the same directory as the batch_evaluator.py script."
        )
        scenarios_to_evaluate = [
            {
                "scenario_id": "full_session_amahle_001",
                "description": "A complete, hybrid evaluation for Amahle (Persona 7), a university student from KZN, who answers some questions conversationally.",
                "notes_for_human_review": "Verify the model can extract multiple data points from a single utterance (turn 3). Check how the assessment handles a direct refusal to answer (turn 6).",
                # The user_persona provides rich context for the test case.
                "user_persona": {
                    "id": "user_persona_07",
                    "age": 21,
                    "details": {
                        "location_description": "Durban, KwaZulu-Natal",
                        "relationship_status_persona": "In a relationship",
                        "education_level_persona": "Currently at university",
                        "num_children_persona": 0,
                    },
                },
                # A single 'turns' array containing the entire, ordered conversation.
                "turns": [
                    {
                        "flow_type": "onboarding",
                        "question_number": 1,
                        "llm_utterance": "Welcome! To get started, which province do you currently reside in? 🏞️",
                        "user_utterance": "Hi, I live in KZN.",
                        "ground_truth_delta": {"province": "KwaZulu-Natal"},
                    },
                    {
                        "flow_type": "onboarding",
                        "question_number": 4,
                        "llm_utterance": "Thanks! What is your highest level of education? 📚",
                        "user_utterance": "I finished my matric last year, now I'm at uni.",
                        "ground_truth_delta": {
                            "education_level": "More than high school"
                        },
                    },
                    {
                        "flow_type": "onboarding",
                        "question_number": 3,
                        "llm_utterance": "Okay, great. What is your current relationship status? 👤",
                        "user_utterance": "I'm in a relationship and I don't have any kids yet.",
                        "ground_truth_delta": {
                            "relationship_status": "In a relationship",
                            "num_children": "0",
                        },
                    },
                    {
                        "flow_type": "assessment",
                        "question_number": 1,
                        "llm_utterance": "How confident are you in making decisions about your health?",
                        "user_utterance": "Very! A 5 for sure.",
                        "ground_truth_delta": {
                            "processed_user_response": "Very confident",
                            "current_assessment_step": 1,
                        },
                    },
                    {
                        "flow_type": "assessment",
                        "question_number": 2,
                        "llm_utterance": "How confident are you discussing medical problems with your doctor or nurse?",
                        "user_utterance": "Uhm, maybe a 3?",
                        "ground_truth_delta": {
                            "processed_user_response": "Somewhat confident",
                            "current_assessment_step": 2,
                        },
                    },
                    {
                        "flow_type": "assessment",
                        "question_number": 3,
                        "llm_utterance": "I feel confident questioning my healthcare provider about my treatment.",
                        "user_utterance": "I'd rather not say for that one.",
                        "ground_truth_delta": {
                            "processed_user_response": "skipped",
                            "current_assessment_step": 3,
                        },
                    },
                ],
            },
        ]
    else:
        scenarios_to_evaluate = load_golden_dataset(DATASET_PATH)

    run_batch_evaluation(scenarios_to_evaluate)
    logger.info("Batch evaluation script completed successfully.")
