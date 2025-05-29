import logging
import json

import doc_store
import pipelines


logger = logging.getLogger(__name__)

#TODO: Add confirmation outputs telling the user what we understood their response to be, before sending them a next message.

def run_simulation():
    """
    Runs a simulation of the onboarding and assessment process using the pipelines.
    """
    logger.info("--- Starting Haystack POC Simulation ---")

    # --- Simulation ---
    # ** Onboarding Scenario **
    logger.info("\n--- Simulating Onboarding ---")
    onboarding_flow_id = "onboarding"
    user_context = { # Simulation data collected progressively
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
        "other": {}
    }
    max_onboarding_steps = 10 # Safety break
    chat_history = []

    # Simulate Onboarding
    for attempt in range(max_onboarding_steps):
        print("-" * 20)
        logger.info(f"Onboarding Question Attempt: {attempt + 1}")

        ### Endpoint 1: The below requires a future API endpoint that Turn can call to get the next question to send to a user.

        # Get remaining questions
        all_questions = doc_store.ONBOARDING_FLOWS.get(onboarding_flow_id, [])
        remaining_questions_list = doc_store.get_remaining_onboarding_questions(user_context, all_questions)

        if not remaining_questions_list:
            logger.info("All onboarding questions answered. Onboarding complete.")
            break

        # LLM decides the next question
        logger.info("Running next question selection pipeline...")
        next_question_pipeline = pipelines.create_next_onboarding_question_pipeline()
        next_question_result = next_question_pipeline.run({
            "prompt_builder": {
                "user_context": user_context,
                "remaining_questions": remaining_questions_list,
                "chat_history": chat_history
            }
        })

        chosen_question_number = None
        if next_question_result and "llm" in next_question_result and next_question_result["llm"]["replies"]:
            try:
                reply_str = next_question_result["llm"]["replies"][0]
                chosen_data = json.loads(reply_str)
                chosen_question_number = chosen_data.get("chosen_question_number")
                contextualized_question = chosen_data.get("contextualized_question")
                logger.info(f"LLM chose question with question_number: {chosen_question_number}")
                logger.info(f"LLM contextualized question: {contextualized_question}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM choice for next question: {reply_str} - Error: {e}")
                # Fallback: pick the first remaining question in the list if LLM fails
                if remaining_questions_list:
                    chosen_question_number = remaining_questions_list[0]['question_number']
                    contextualized_question = remaining_questions_list[0]['content']
                    logger.warning(f"Falling back to first remaining question: question_number {chosen_question_number}")
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error processing LLM choice: {e}")
                if remaining_questions_list:
                    chosen_question_number = remaining_questions_list[0]['question_number']
                    contextualized_question = remaining_questions_list[0]['content']
                    logger.warning(f"Falling back to first remaining question due to error: question_number {chosen_question_number}")
        if chosen_question_number is None:
            logger.error("Could not determine next question. Ending onboarding.")
            break
        
        ### Endpoint 1 can return the next question to ask the user. The code below is just for simulation, to compare 
        ### the contextualized question with the original question.

        # Find the actual question data using the chosen_question_number
        current_step_data = next((q for q in all_questions if q['question_number'] == chosen_question_number), None)

        if not current_step_data:
            logger.error(f"Could not find question data for question_number {chosen_question_number}. Skipping.")
            continue
        logger.info(f"Original question: '{current_step_data['content']}' (collects: {current_step_data['collects']})")

        ### Endpoint 2: The below requires a future API endpoint that Turn can call to get extracted data from a user response.
        ### The chat history and terminal interaction with the user is just for the simulation's sake.

        # Simulate User Response & Data Extraction
        chat_history.append("System to User: " + contextualized_question+ "\n> ")
        user_response = input(contextualized_question + "\n> ")

        logger.info("Running data extraction pipeline...")
        onboarding_data_extraction_pipe = pipelines.create_onboarding_data_extraction_pipeline()
        extraction_result = onboarding_data_extraction_pipe.run({
            "prompt_builder": {
                "user_response": user_response,
                "user_context": user_context,
                "chat_history": chat_history
                }
        })
        chat_history.append("User to System: " + user_response+ "\n> ")

        if extraction_result and "llm" in extraction_result and extraction_result["llm"]["replies"]:
            extracted_json_str = extraction_result["llm"]["replies"][0]
            logger.info(f"Raw extraction result: {extracted_json_str}")
            try:
                extracted_data = json.loads(extracted_json_str)
                print(f"[Extracted Data]:\n{json.dumps(extracted_data, indent=2)}\n")
                
                # Try to collect/store extracted data:
                onboarding_data_to_collect = [
                    "province", "area_type", "relationship_status", "education_level",
                    "hunger_days", "num_children", "phone_ownership"
                ]
                for k, v in extracted_data.items():
                    logger.info(f"Extracted {k}: {v}")
                    if k in onboarding_data_to_collect:
                        user_context[k] = v
                        logger.info(f"Updated user_context for {k}: {v}")
                    else:
                        user_context["other"][k] = v

                logger.info(f"Updated user_context: {user_context}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from extraction result: {extracted_json_str}")
        else:
             logger.warning("Data extraction pipeline did not produce a result.")
        
        ### Endpoint 2 can maybe return the updated user_context dictionary such that Turn can update the user profile accordingly.

    if not remaining_questions_list:
        logger.info("Successfully completed dynamic onboarding flow.")
    else:
        logger.warning(f"Onboarding flow ended with {len(remaining_questions_list)} questions remaining after {max_onboarding_steps} attempts.")


    # ** Assessment Scenario **
    print("")
    logger.info("\n--- Simulating Assessment ---")
    assessment_flow_id = "dma-assessment"
    current_assessment_step = 0
    user_context['goal'] = "Complete the assessment"
    all_questions = doc_store.ASSESSMENT_FLOWS.get(assessment_flow_id, [])
    max_assessment_steps = len(all_questions)*2 # Safety break

    # Simulate Assessment
    while current_assessment_step < max_assessment_steps:
        print("-" * 20)
        logger.info(f"Assessment Step: Requesting step after {current_assessment_step}")

        ### Endpoint 3: The below requires a future API endpoint that Turn can call to get an assessment question to send to the user.
        ### Some of the logic below is just for simulation, e.g. the while loop itself.

        # Get the next raw step data
        next_step_data = all_questions[current_assessment_step+1]

        if not next_step_data:
            logger.info("Assessment flow complete.")
            break

        current_question_number = next_step_data['question_number']
        logger.info(f"Processing step {current_question_number} for flow '{assessment_flow_id}'")

        # Contextualize the current question
        logger.info("Running contextualization pipeline...")
        retriever_filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.flow_id", "operator": "==", "value": assessment_flow_id},
                {"field": "meta.question_number", "operator": "==", "value": current_question_number},
            ]
        }
        assessment_contextualization_pipe = pipelines.create_assessment_contextualization_pipeline()
        contextualization_result = assessment_contextualization_pipe.run({
            "retriever": {"filters": retriever_filters},
            "prompt_builder": {
                "user_context": user_context
            }
        })

        if contextualization_result and "llm" in contextualization_result and contextualization_result["llm"]["replies"]:
            contextualized_question = contextualization_result["llm"]["replies"][0]
        else:
            # Fallback
            logger.warning(f"Contextualization failed for step {current_question_number}. Using raw content.")
            contextualized_question = next_step_data['content']
            print(f"\n[System to User (Fallback)]:\n{contextualized_question}\n")

        ### Endpoint 3 can return the contextualized_question such that Turn can send it to the user.

        # Simulate User Response
        user_response = input(contextualized_question + "\n> ")
        
        ### Endpoint 4: The below requires a future API endpoint that Turn can call to validate a user's response to an assessment question.

        # Validate User Response
        validator_pipe = pipelines.create_assessment_response_validator_pipeline()
        validation_result = validator_pipe.run({
            "prompt_builder": {
                "user_response": user_response
            }
        })

        if validation_result and "llm" in validation_result and validation_result["llm"]["replies"]:
            processed_user_response = contextualization_result["llm"]["replies"][0]
            print(f"\n[Processed user response]:\n{processed_user_response}\n")
        else:
            # Fallback
            logger.warning(f"Response validation failed for step {current_question_number}.")
            processed_user_response = "nonsense"

        # Move to the next step, or try again if the response was invalid
        if processed_user_response != "nonsense":
            logger.info(f"Storing validated response for question_number {current_question_number}: {processed_user_response}")
            current_assessment_step = current_question_number
        
        ### Endpoint 4 can return the processed_user_response such that Turn can know if the response was valid or not. Thereafter, Turn
        ### can call Endpoint 3 again to either try the same question again, or move on to the next if the response was valid.

    logger.info("--- Simulation Complete ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_simulation()
