# pipelines.py
import json
import logging
import os
from functools import cache
from typing import Any, Dict, List, Optional # Python 3.9+ dict, list

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import FilterRetriever # Ensure correct import if used
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage, Document # Import Document
from haystack.tools import Tool
from haystack.utils import Secret

# Assuming doc_store.py is in the same package and provides setup_document_store
from .doc_store import setup_document_store

logger = logging.getLogger(__name__)


USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "true").lower() == "true"  # TOBE updated to false in production

print(f"USE_MOCK_LLM is set to {USE_MOCK_LLM}. This will affect LLM interactions.")

NEXT_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_question_number": {"type": "integer"},
        "contextualized_question": {"type": "string"},
    },
    "required": ["chosen_question_number", "contextualized_question"],
    "additionalProperties": False, # Usually false for strict schema
}

# --- Mock Data Examples ---
MOCK_ONBOARDING_QUESTIONS_CYCLE = [
    {"chosen_question_number": 1, "contextualized_question": "Mocked: Province?", "collects_field": "province", "fallback_used": False},
    {"chosen_question_number": 2, "contextualized_question": "Mocked: Area type?", "collects_field": "area_type", "fallback_used": False},
    {"chosen_question_number": 3, "contextualized_question": "Mocked: Relationship status?", "collects_field": "relationship_status", "fallback_used": False},
    {"chosen_question_number": 4, "contextualized_question": "Mocked: Education?", "collects_field": "education_level", "fallback_used": False},
    {"chosen_question_number": 5, "contextualized_question": "Mocked: Hunger days?", "collects_field": "hunger_days", "fallback_used": False},
    {"chosen_question_number": 6, "contextualized_question": "Mocked: Num children?", "collects_field": "num_children", "fallback_used": False},
    {"chosen_question_number": 7, "contextualized_question": "Mocked: Phone ownership?", "collects_field": "phone_ownership", "fallback_used": False},
]
mock_onboarding_question_index = 0

MOCK_ASSESSMENT_QUESTIONS_CONTENT = {
    "dma-assessment": {
        1: "Mock Assessment Q1: How confident are you in making health decisions?",
        # Add more for your assessment flow
    }
}

def get_llm_generator() -> Optional[OpenAIChatGenerator]: # REMOVED @cache
    if USE_MOCK_LLM and not os.getenv("OPENAI_API_KEY"): # If mocking and no key, don't even try
        logger.info("Mock mode active and no OpenAI key; LLM generator not created.")
        return None
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        llm_generator = OpenAIChatGenerator(api_key=openai_api_key, model=llm_model)
        logger.info(f"OpenAI Chat Generator instance created for model {llm_model}.")
        return llm_generator
    except ValueError:
        logger.warning("OPENAI_API_KEY not found. LLM Generator not created (this is OK if USE_MOCK_LLM=true).")
        return None

def extract_onboarding_data(**kwargs) -> dict[str, Any]:
    logger.debug(f"Tool 'extract_onboarding_data' called with: {kwargs}")
    return kwargs

@cache
def create_onboarding_tool() -> Tool:
    # ... (Your full tool definition from before)
    tool = Tool(
        name="extract_onboarding_data",
        description="Extract structured data points...",
        function=extract_onboarding_data,
        parameters={
            "type": "object",
            "properties": {
                "province": {"type": "string", "description": "The user's province.", "enum": ["Eastern Cape","Free State","Gauteng","KwaZulu-Natal","Limpopo","Mpumalanga","Northern Cape","North West","Western Cape"]},
                "area_type": {"type": "string", "description": "The type of area the user lives in.", "enum": ["City","Township or suburb","Town","Farm or smallholding","Village","Rural area"]},
                "relationship_status": {"type": "string", "description": "The user's relationship status.", "enum": ["Single","Relationship","Married","Skip"]},
                "education_level": {"type": "string", "description": "The user's highest education level.", "enum": ["No school","Some primary","Finished primary","Some high school","Finished high school","More than high school","Don't know","Skip"]},
                "hunger_days": {"type": "string", "description": "Number of days in the past 7 days the user didn't have enough to eat.", "enum": ["0 days","1-2 days","3-4 days","5-7 days"]},
                "num_children": {"type": "string", "description": "The number of children the user has.", "enum": ["0","1","2","3","More than 3","Why do you ask?"]},
                "phone_ownership": {"type": "string", "description": "Whether the user owns their phone.", "enum": ["Yes","No","Skip"]}
            },
            "additionalProperties": {"type": "string", "description": "Any other valuable maternal health-related information extracted."}
        }
    )
    return tool

@cache
def create_next_onboarding_question_pipeline() -> Optional[Pipeline]:
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Next Onboarding Q Pipeline.")
        return None

    pipeline = Pipeline()
    prompt_template = """
    You are an assistant helping to onboard a new user.
    User Context:
    {% for key, value in user_context.items() %}- {{ key }}: {{ value if value is not none else 'Not provided' }}
    {% endfor %}
    Chat History (last few messages):
    {% for message in chat_history[-6:] %}- {{ message }}
    {% endfor %}
    Remaining questions (DO NOT ask questions already covered in context):
    {% for q in remaining_questions %}- Q#: {{ q.question_number }}, Asks for: "{{ q.collects }}", Example Q: "{{ q.content }}"
    {% endfor %}
    Choose the best single question number from 'Remaining questions' to ask next.
    Contextualize the chosen question naturally.
    Response MUST be valid JSON: {"chosen_question_number": int, "contextualized_question": str}
    JSON Response:""" # Simplified prompt for brevity
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "remaining_questions", "chat_history"]
    )
    json_validator = JsonSchemaValidator(json_schema=NEXT_QUESTION_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator: # Only add if available (might be None if API key missing, even in mock if create fails)
        pipeline.add_component("llm", llm_generator)
        pipeline.add_component("json_validator", json_validator)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "json_validator.messages")
    else:
        logger.info("Next Onboarding Question Pipeline created without LLM (likely mock mode or no API key).")
    logger.info("Created Next Question Selection Pipeline.")
    return pipeline

@cache
def create_onboarding_data_extraction_pipeline() -> Optional[Pipeline]:
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Onboarding Data Extraction Pipeline.")
        return None

    pipeline = Pipeline()
    # Using the detailed prompt from your original file
    prompt_template = """
    You are helping extract onboarding data from a user's response to a maternal health chatbot.
    Data already collected:
    {{ user_context }}
    Chat history:
    {{ chat_history }}
    User's latest message: "{{ user_response }}"
    Please use the 'extract_onboarding_data' tool to analyze this message. Pay close attention to these **critical rules**:
    - For the properties 'province', 'area_type', 'relationship_status', 'education_level', 'hunger_days', 'num_children' and 'phone_ownership', extracted data **MUST** adhere strictly to their 'enum' lists. If the user's response for one of these properties does **NOT** contain a word or phrase that *directly and unambiguously* maps to one of the EXACT 'enum' values, **DO NOT include that property in your tool call**. Only store the 'Skip' enum value for these properties if the user explicitly states they want to skip in response to that specific question.
    - **DO NOT GUESS or INFER** an enum value based on sentiment, vague descriptions, or ambiguous terms. Only include a field if you are highly confident that the user's input matches an allowed 'enum' value.
    - Do not extract a data point if it clearly has already been collected in the user context, unless the user explicitly provides new information that updates it.
    - For 'hunger_days' and 'num_children', if the user provides a number that does not match any of the enum values, you **MUST** omit those fields, unless the corresponding enum can reasonably be deduced or inferred e.g. '6' hungry days can be mapped to the enum '5-7 days'.
    - For the open-ended additionalProperties, extract any extra information mentioned AS LONG AS it pertains specifically to maternal health or the use of a maternal health chatbot.
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "remaining_questions", "chat_history"]
    )
    pipeline.add_component("prompt_builder", prompt_builder)

    if llm_generator:
        extraction_tool = create_onboarding_tool()
        # Create a new llm_generator instance or ensure tools are set on a fresh one if get_llm_generator is cached
        # Since get_llm_generator is NOT cached anymore, this is fine.
        llm_generator_for_tool = get_llm_generator() # Get a potentially fresh instance for tool use
        if llm_generator_for_tool:
            llm_generator_for_tool.tools = [extraction_tool]
            pipeline.add_component("llm", llm_generator_for_tool)
            pipeline.connect("prompt_builder.prompt", "llm.messages")
        else:
            logger.warning("LLM generator not available for tool use in extraction pipeline (OK if mocking).")
    logger.info("Created Onboarding Data Extraction Pipeline.")
    return pipeline

@cache
def create_assessment_contextualization_pipeline() -> Optional[Pipeline]:
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Assessment Contextualization Pipeline.")
        return None
        
    pipeline = Pipeline()
    # Using the detailed prompt from your original file
    prompt_template = """
    You are an assistant helping to personalize assessment questions on a maternal health chatbot service.
    User Context:
    {% for key, value in user_context.items() %}- {{ key }}: {{ value if value is not none else 'Not provided' }}
    {% endfor %}
    Review the following assessment question intended for sequence step {{ documents[0].meta.question_number }}.
    If you think it's needed, make minor adjustments to ensure that the question is clear and directly applicable to the user's context.
    **Crucially, do not change the core meaning, difficulty, or the scale/format of the question.**
    Just ensure clarity and relevance. If no changes are needed, return the original question.
    Make sure that the list of valid responses is at the end of the contextualized question.
    Original Assessment Question: {{ documents[0].content }}
    Valid Responses: 1 - Not at all confident, 2 - A little confident, 3 - Somewhat confident, 4 - Confident, 5 - Very confident
    Contextualized Question:"""
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "remaining_questions", "chat_history"]
    )
    
    # Document store setup should be robust
    document_store = setup_document_store()
    if not document_store:
        logger.error("Document store not available. Cannot create Assessment Contextualization Pipeline.")
        return None
    retriever = FilterRetriever(document_store=document_store)

    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator:
        pipeline.add_component("llm", llm_generator)
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
    logger.info("Created Assessment Contextualization Pipeline.")
    return pipeline

@cache
def create_assessment_response_validator_pipeline() -> Optional[Pipeline]:
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Assessment Response Validation Pipeline.")
        return None

    pipeline = Pipeline()
    # Using the detailed prompt from your original file
    prompt_template = """
    You are an assistant helping to validate user responses to an assessment question on a maternal health chatbot service.
    The valid possible responses are: 1 - Not at all confident, 2 - A little confident, 3 - Somewhat confident, 4 - Confident, 5 - Very confident
    User responses are unpredictable. They might reply with a number, or with text, or with both, corresponding to a valid response. Or, they might respond with nonsense or gibberish.
    If you think that the user response maps to one of the valid responses, your output must be the text corresponding to that valid response (i.e. without the number).
    If you think that the user response is nonsense or gibberish instead, return "nonsense".
    User Response: {{ user_response }}
    Validated Response:"""
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "remaining_questions", "chat_history"]
    )
    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator:
        pipeline.add_component("llm", llm_generator)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
    logger.info("Created Assessment Response Validation Pipeline.")
    return pipeline

# --- Running Pipelines (with Mocking Logic and Corrected Returns) ---

def run_next_onboarding_question_pipeline(
    pipeline: Optional[Pipeline],
    user_context: dict[str, Any],
    remaining_questions: list[dict[str, Any]],
    chat_history: list[str]
) -> Optional[dict[str, Any]]:
    global mock_onboarding_question_index
    if USE_MOCK_LLM:
        logger.info("MOCKING: run_next_onboarding_question_pipeline")
        if not remaining_questions: return None
        mock_response = None
        for i in range(len(MOCK_ONBOARDING_QUESTIONS_CYCLE)): # Try to find a relevant mock
            current_mock_idx = (mock_onboarding_question_index + i) % len(MOCK_ONBOARDING_QUESTIONS_CYCLE)
            potential_mock = MOCK_ONBOARDING_QUESTIONS_CYCLE[current_mock_idx]
            if any(q["question_number"] == potential_mock["chosen_question_number"] for q in remaining_questions):
                mock_response = {**potential_mock, "fallback_used": False} # Ensure all fields
                mock_onboarding_question_index = (current_mock_idx + 1) % len(MOCK_ONBOARDING_QUESTIONS_CYCLE)
                break
        if not mock_response: # Fallback mock
            q_def = remaining_questions[0]
            mock_response = {
                "chosen_question_number": q_def["question_number"],
                "contextualized_question": f"Mocked Fallback: {q_def['content']}",
                "collects_field": q_def.get("collects"),
                "fallback_used": True
            }
        logger.info(f"Mocked next question response: {mock_response}")
        return mock_response

    if not pipeline or not get_llm_generator(): # Handles case where API key might be missing, and not mocking
        logger.warning("Actual LLM pipeline not available for next onboarding question. Using fallback.")
        if remaining_questions:
            q_def = remaining_questions[0]
            return {"chosen_question_number": q_def['question_number'],
                    "contextualized_question": q_def['content'],
                    "collects_field": q_def.get("collects"), "fallback_used": True}
        return None

    logger.info("CALLING ACTUAL LLM: run_next_onboarding_question_pipeline")
    try:
        # CORRECTED pipeline.run call
        result = pipeline.run(data={
            "user_context": user_context,
            "remaining_questions": remaining_questions,
            "chat_history": chat_history
        })
        validated_responses = result.get("json_validator", {}).get("validated", [])
        if validated_responses and isinstance(validated_responses[0], ChatMessage):
            chosen_data_str = validated_responses[0].content # Use .content for ChatMessage
            chosen_data = json.loads(chosen_data_str)
            chosen_q_num = chosen_data.get("chosen_question_number")
            contextualized_q = chosen_data.get("contextualized_question")
            collects_field = None
            if chosen_q_num is not None:
                for q_def in remaining_questions:
                    if q_def.get("question_number") == chosen_q_num:
                        collects_field = q_def.get("collects")
                        break
            logger.info(f"LLM chose question_number: {chosen_q_num}, collects: {collects_field}")
            return {"chosen_question_number": chosen_q_num,
                    "contextualized_question": contextualized_q,
                    "collects_field": collects_field, "fallback_used": False}
        else:
            logger.warning("LLM failed to produce valid JSON. Using fallback.")
    except Exception as e:
        logger.error(f"Error in next_onboarding_question_pipeline: {e}", exc_info=True)

    if remaining_questions: # Fallback for actual LLM path if error or no valid JSON
        q_def = remaining_questions[0]
        logger.warning(f"Falling back to first remaining question (actual path): question_number {q_def['question_number']}")
        return {"chosen_question_number": q_def['question_number'],
                "contextualized_question": q_def['content'],
                "collects_field": q_def.get("collects"), "fallback_used": True}
    return None


def run_onboarding_data_extraction_pipeline(
    pipeline: Optional[Pipeline],
    user_response: str,
    user_context: dict[str, Any],
    chat_history: list[str],
    expected_collects_field: Optional[str] = None # NEW PARAMETER
) -> dict[str, Any]:
    if USE_MOCK_LLM:
        logger.info(f"MOCKING: run_onboarding_data_extraction_pipeline, expecting field: {expected_collects_field}")
        mock_extracted_data = {}

        if expected_collects_field == "province":
            # Simulate extracting from a phrase like "I'm in the Free State near Bloem"
            if "free state" in user_response.lower(): mock_extracted_data["province"] = "Free State"
            elif "gauteng" in user_response.lower(): mock_extracted_data["province"] = "Gauteng"
            else: mock_extracted_data["province"] = "Mocked Western Cape" # Default mock for province
        elif expected_collects_field == "area_type":
            if "city" in user_response.lower(): mock_extracted_data["area_type"] = "City"
            elif "farm" in user_response.lower(): mock_extracted_data["area_type"] = "Farm or smallholding"
            else: mock_extracted_data["area_type"] = "Mocked Township or suburb"
        elif expected_collects_field == "relationship_status":
            if "single" in user_response.lower(): mock_extracted_data["relationship_status"] = "Single"
            elif "married" in user_response.lower(): mock_extracted_data["relationship_status"] = "Married"
            else: mock_extracted_data["relationship_status"] = "Mocked Relationship"
        elif expected_collects_field == "education_level":
            if "high school" in user_response.lower(): mock_extracted_data["education_level"] = "Finished high school"
            elif "primary" in user_response.lower(): mock_extracted_data["education_level"] = "Finished primary"
            else: mock_extracted_data["education_level"] = "Mocked Some high school"
        elif expected_collects_field == "hunger_days":
            if "0 days" in user_response or "never" in user_response.lower(): mock_extracted_data["hunger_days"] = "0 days"
            elif "1" in user_response or "2" in user_response: mock_extracted_data["hunger_days"] = "1-2 days"
            else: mock_extracted_data["hunger_days"] = "Mocked 3-4 days"
        elif expected_collects_field == "num_children":
            if "0" in user_response or "none" in user_response.lower(): mock_extracted_data["num_children"] = "0"
            elif "1" in user_response or "one" in user_response.lower(): mock_extracted_data["num_children"] = "1"
            else: mock_extracted_data["num_children"] = "Mocked 2"
        elif expected_collects_field == "phone_ownership":
            if "yes" in user_response.lower() or "own it" in user_response.lower(): mock_extracted_data["phone_ownership"] = "Yes"
            elif "no" in user_response.lower() or "share" in user_response.lower(): mock_extracted_data["phone_ownership"] = "No"
            else: mock_extracted_data["phone_ownership"] = "Mocked Skip"
        else:
            # Fallback if field is unknown or not specifically mocked yet
            # You could make this even smarter by having a generic response
            logger.warning(f"No specific mock logic for expected_collects_field: '{expected_collects_field}'. User response: '{user_response}'")
            mock_extracted_data[f"mock_other_{expected_collects_field or 'unknown'}"] = f"Extracted based on '{user_response}'"

        # Example: Simulate extracting an 'other' field if certain keywords appear
        if "pregnant" in user_response.lower() and "worried" in user_response.lower():
            mock_extracted_data["additional_symptom_or_concern"] = "User mentioned being worried during pregnancy."
        
        logger.info(f"Mocked extracted data: {mock_extracted_data}")
        return mock_extracted_data

    # --- Actual LLM call logic (remains the same) ---
    if not pipeline or not get_llm_generator(): # Guard clause
        logger.warning("Actual LLM pipeline not available for data extraction. Returning empty dict.")
        return {}
        
    logger.info(f"CALLING ACTUAL LLM: run_onboarding_data_extraction_pipeline (expecting data for: {expected_collects_field or 'any field'})")
    try:
        result = pipeline.run(data={
            "user_response": user_response,
            "user_context": user_context,
            "chat_history": chat_history
        })
        llm_response = result.get("llm", {})
        replies = llm_response.get("replies", [])
        if replies and isinstance(replies[0], ChatMessage):
            first_reply = replies[0]
            tool_calls = first_reply.tool_calls or []
            if tool_calls and isinstance(tool_calls[0], dict):
                arguments_str = tool_calls[0].get('arguments', '{}')
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    if isinstance(arguments, dict):
                        logger.info(f"LLM extracted data (tool arguments): {arguments}")
                        return arguments
                    else:
                        logger.warning(f"Tool arguments parsed but not a dictionary: {arguments}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments JSON: {arguments_str}")
            else:
                logger.info("No tool calls invoked by LLM for data extraction.")
        else:
            logger.warning("No valid replies in LLM response for data extraction.")
        return {}
    except Exception as e:
        logger.error(f"Error running extraction pipeline: {e}", exc_info=True)
        return {}


def run_assessment_contextualization_pipeline(
    pipeline: Optional[Pipeline], flow_id: str, question_number: int, user_context: dict[str, Any]
) -> Optional[str]:
    if USE_MOCK_LLM:
        logger.info("MOCKING: run_assessment_contextualization_pipeline")
        raw_q_content = MOCK_ASSESSMENT_QUESTIONS_CONTENT.get(flow_id, {}).get(question_number, f"Default Mock Q{question_number}")
        mock_q = f"Mocked (Q{question_number}): Considering user age {user_context.get('age', 'N/A')}, {raw_q_content}"
        logger.info(f"Mocked contextualized question: {mock_q}")
        return mock_q

    if not pipeline or not get_llm_generator():
        logger.warning("Actual LLM pipeline for assessment contextualization not available. Attempting raw content.")
        # Fallback to just getting the document if pipeline or LLM failed to init
        doc_store = setup_document_store()
        if doc_store:
            try:
                retriever = FilterRetriever(document_store=doc_store)
                filters = {"operator": "AND", "conditions": [
                    {"field": "meta.flow_id", "operator": "==", "value": flow_id},
                    {"field": "meta.question_number", "operator": "==", "value": question_number}]}
                retrieved_docs = retriever.run(filters=filters).get("documents", [])
                if retrieved_docs and isinstance(retrieved_docs[0], Document): return retrieved_docs[0].content
            except Exception as e_fb: logger.error(f"Fallback retriever failed: {e_fb}")
        return None # Could not retrieve raw question

    logger.info("CALLING ACTUAL LLM: run_assessment_contextualization_pipeline")
    try:
        filters = {"operator": "AND", "conditions": [
            {"field": "meta.flow_id", "operator": "==", "value": flow_id},
            {"field": "meta.question_number", "operator": "==", "value": question_number}]}
        # CORRECTED pipeline.run call structure
        result = pipeline.run(data={
            "retriever": {"filters": filters}, # Data for retriever component
            "user_context": user_context      # Data for prompt_builder component
        })
        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies") and isinstance(llm_response["replies"][0], ChatMessage):
            return llm_response["replies"][0].content
        logger.warning("No valid replies in LLM response for contextualization")
        return None
    except Exception as e:
        logger.error(f"Error in assessment_contextualization_pipeline: {e}", exc_info=True)
        return None


def run_assessment_response_validator_pipeline(
    pipeline: Optional[Pipeline], user_response: str
) -> Optional[str]:
    if USE_MOCK_LLM:
        logger.info("MOCKING: run_assessment_response_validator_pipeline")
        # More sophisticated mock could use valid_responses_options if passed
        processed = f"MockValidated: {user_response.strip().capitalize()}"
        if "nonsense" in user_response.lower() or len(user_response) < 3: # Basic heuristic for mock
            processed = "nonsense"
        logger.info(f"Mocked validated response: {processed}")
        return processed

    if not pipeline or not get_llm_generator():
        logger.warning("Actual LLM pipeline for validation not available. Passing through response.")
        return user_response # Passthrough if no LLM

    logger.info("CALLING ACTUAL LLM: run_assessment_response_validator_pipeline")
    try:
        # CORRECTED pipeline.run call
        result = pipeline.run(data={"user_response": user_response})
        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies") and isinstance(llm_response["replies"][0], ChatMessage):
            return llm_response["replies"][0].content
        logger.warning("No valid replies in LLM response for validation")
        return None # Or user_response as a fallback
    except Exception as e:
        logger.error(f"Error in assessment_response_validator_pipeline: {e}", exc_info=True)
        return None
