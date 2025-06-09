import logging
import datetime
import copy
from . import tasks
from langfuse import Langfuse


logger = logging.getLogger(__name__)

# TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.


def run_simulation():
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    Then logs data to Langfuse using dynamic trace names
    based on question numbers for robust, turn-by-turn evaluation.
    """
    # --- LANGFUSE: Initialize the client ---
    langfuse = Langfuse()
    session_id = f"interactive-session-{datetime.datetime.now().isoformat()}"
    
    logger.info("--- Starting Haystack POC Simulation  (Session: {session_id}) ---")

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

    # Simulate Onboarding
    for attempt in range(max_onboarding_steps):
        print("-" * 20)
        logger.info(f"Onboarding Question Attempt: {attempt + 1}")

        # --- UPDATED LOGIC 1: Handle the dictionary returned by the task ---
        # We now expect a dictionary containing the question and its number.
        question_data = tasks.get_next_onboarding_question(
            user_context, chat_history
        )

        # Add a robust check in case the task returns None or an incomplete dictionary.
        if not question_data or not question_data.get("question_number"):
            logger.info("Onboarding flow complete or invalid data returned from task.")
            break

        # Unpack the dictionary into the variables we need.
        contextualized_question = question_data.get("question", "Error: No question text found.")
        question_number = question_data.get("question_number")

        # --- UPDATED LOGIC 2: Create the trace with the dynamic name ---
        context_before = copy.deepcopy(user_context)
        trace = langfuse.trace(
            name=f"onboarding-q{question_number}",
            session_id=session_id,
            metadata={"initial_context": context_before}
        )

        # Simulate User Response & Data Extraction
        chat_history.append({"role": "system", "content": contextualized_question})
        user_response = input(contextualized_question + "\n> ")
        chat_history.append({"role": "user", "content": user_response})

        # --- LANGFUSE: Update trace ---
        trace.update(
            input={
                "question": contextualized_question,
                "user_response": user_response
            }
        )

        ### Endpoint 2: Turn can call to get extracted data from a user response.
        # --- LANGFUSE: Create a "generation" for the LLM extraction step ---
        generation_extract = trace.generation(
            name="extract-onboarding-data",
            input=user_response,
            model="your-extraction-model",  # IMPORTANT: Replace with your actual model name
        )

        user_context = tasks.extract_onboarding_data_from_response(
            user_response, user_context, chat_history
        )
        # --- LANGFUSE: Find what was newly extracted and log it as the output ---
        newly_extracted_data = {
            k: user_context[k] for k in user_context
            if user_context.get(k) != context_before.get(k)
        }
        generation_extract.end(output={"extracted_data": newly_extracted_data})

        # --- LANGFUSE: Update the trace's final output to be the new state ---
        # The trace's final output is now the *delta* (what's new), plus the
        # field that was being collected in this turn.
        trace.update(
            output={
                **newly_extracted_data,  # Unpack the newly extracted data
            }
        )

    # ** Assessment Scenario **
    print("")
    logger.info("\n--- Simulating Assessment ---")
    current_assessment_step = 0
    user_context["goal"] = "Complete the assessment"
    max_assessment_steps = 10  # Safety break

    # Simulate Assessment
    while current_assessment_step < max_assessment_steps:
        print("-" * 20)
        logger.info(f"Assessment Step: Requesting step after {current_assessment_step}")

        # --- UPDATED LOGIC 3: Apply the same dynamic naming to the assessment part ---
        result_q = tasks.get_assessment_question(
            flow_id="dma_flow_id",
            question_number=current_assessment_step,
            current_assessment_step=current_assessment_step,
            user_context=user_context,
        )

        if not result_q or not result_q.get("current_question_number"):
            logger.info("Assessment flow complete.")
            break

        # Unpack the results
        contextualized_question = result_q["contextualized_question"]
        question_number = result_q["current_question_number"] # This is the step number


        # --- LANGFUSE: Capture the "before" state for the assessment turn ---
        # --- LANGFUSE: Create a trace for the assessment turn ---
        context_before = copy.deepcopy(user_context)
        trace = langfuse.trace(
            name=f"assessment-q{question_number}",
            session_id=session_id,
            metadata={"initial_context": context_before}
        )

        ### Endpoint 3: Turn can call to get an assessment question to send to the user.
        # --- LANGFUSE: Create a "span" for this step ---
        trace.span(name="get-assessment-question", output=result_q)

        # Simulate User Response
        user_response = input(contextualized_question + "\n> ")

        # --- LANGFUSE: Update trace with user input and bot question ---
        trace.update(
            input={
                "question": contextualized_question,
                "user_response": user_response
            }
        )

        ### Endpoint 4: Turn can call to validate a user's response to an assessment question.
        # --- LANGFUSE: Create a "generation" for the LLM validation step ---
        generation_validate = trace.generation(
            name="validate-assessment-answer",
            input=user_response,
            model="your-validation-model"  # IMPORTANT: Replace with your model name
        )

        result_v = tasks.validate_assessment_answer(
            user_response, question_number
        )
        # --- LANGFUSE: Log the output of the generation ---
        # --- LANGFUSE: Update trace with the final result ---
        generation_validate.end(output=result_v)
        trace.update(output=result_v)

        if not result_v:
            logger.warning(
                f"Response validation failed for step {question_number}."
            )
            continue
        # processed_user_response = result['processed_user_response']
        current_assessment_step = result_v["current_assessment_step"]

    logger.info("--- Simulation Complete ---")
    # --- LANGFUSE: Ensure all buffered data is sent to the server before the script exits ---
    langfuse.flush()
    langfuse.shutdown()
    print("All trace data flushed to Langfuse.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_simulation()
