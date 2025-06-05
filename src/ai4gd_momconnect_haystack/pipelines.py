# pipelines.py
"""
Defines and configures Haystack pipelines for onboarding and assessment tasks.

This module includes:
- Configuration for using mock LLM responses via an environment variable.
- JSON schemas for validating LLM outputs.
- Dynamically generated mock data based on flow definitions from doc_store.
- Functions to create and run Haystack pipelines for various chatbot tasks.
"""

import json
import logging
import os
from functools import cache
from typing import Any, Optional # Retaining Optional for explicit None returns

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import FilterRetriever
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage, Document
from haystack.tools import Tool
from haystack.utils import Secret

# Import flow definitions and setup_document_store from doc_store.py
from .doc_store import (
    ONBOARDING_FLOWS, # Loaded dict from onboarding_flow.json
    ASSESSMENT_FLOWS, # Loaded dict from assessment_flow.json
    setup_document_store,
)

logger = logging.getLogger(__name__)

# --- Configuration ---
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

logger.info(
    "PIPELINES.PY: USE_MOCK_LLM is set to %s.", USE_MOCK_LLM
)

# --- JSON Schemas ---
NEXT_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_question_number": {"type": "integer"},
        "contextualized_question": {"type": "string"},
    },
    "required": ["chosen_question_number", "contextualized_question"],
    "additionalProperties": False,
}

# --- Dynamically Populated Mock Data ---

def _create_mock_onboarding_cycle_from_flow(
    flow_data: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Creates mock onboarding questions from the loaded flow definition."""
    mock_cycle: list[dict[str, Any]] = []
    # Assuming tasks.onboarding_flow_id is "onboarding" as used in doc_store.py
    # If tasks.py is importable here, use tasks.onboarding_flow_id
    onboarding_key = "onboarding" # Default key
    try:
        from . import tasks # Try to import tasks to get the key
        onboarding_key = tasks.onboarding_flow_id
    except ImportError:
        logger.warning(
            "Could not import 'tasks' module in pipelines.py for flow_id key; "
            "defaulting to 'onboarding' for mock creation."
        )

    onboarding_questions = flow_data.get(onboarding_key, [])
    if not onboarding_questions:
        logger.warning(
            "Onboarding flow definition ('%s' key) is empty or not found "
            "in provided flow_data for creating mocks. Returning minimal fallback.",
            onboarding_key
        )
        # Minimal fallback if flow data is missing
        return [
            {
                "chosen_question_number": 1,
                "contextualized_question": "Mocked Default: Province?",
                "collects_field": "province",
                "fallback_used": True, # Indicates this is a hardcoded fallback
            }
        ]

    for q_def in onboarding_questions:
        if "question_number" in q_def and "content" in q_def and "collects" in q_def:
            mock_cycle.append({
                "chosen_question_number": q_def["question_number"],
                "contextualized_question": (
                    f"Mocked (dynamic): {q_def['content']}"
                ),
                "collects_field": q_def["collects"],
                "fallback_used": False,
            })
        else:
            logger.warning(
                "Skipping question in mock creation due to missing fields: %s",
                q_def.get("content", "N/A"),
            )
    logger.info(
        "Dynamically created %d mock onboarding questions.", len(mock_cycle)
    )
    return mock_cycle if mock_cycle else [] # Ensure it's a list

def _create_mock_assessment_content_from_flow(
    flow_data: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[int, str]]:
    """Creates mock assessment question content from loaded flow definitions."""
    mock_assessment_data: dict[str, dict[int, str]] = {}
    if not flow_data:
        logger.warning(
            "Assessment flow definition data is empty or not found for "
            "creating mocks. Returning minimal fallback."
        )
        return { # Minimal fallback
            "dma-assessment": {
                1: "Mock Default Assessment Q1: How confident are you?"
            }
        }

    for flow_id, questions in flow_data.items():
        mock_assessment_data[flow_id] = {}
        for q_def in questions:
            if "question_number" in q_def and "content" in q_def:
                mock_assessment_data[flow_id][q_def["question_number"]] = (
                    f"Mocked Assessment (dynamic) Q{q_def['question_number']}: "
                    f"{q_def['content']}"
                )
            else:
                logger.warning(
                    "Skipping assessment question in mock creation due to "
                    "missing fields: %s", q_def.get("content", "N/A")
                )
    logger.info(
        "Dynamically created mock assessment content for %d flows.",
        len(mock_assessment_data)
    )
    return mock_assessment_data if mock_assessment_data else {}

# Populate mocks using the functions and imported flow definitions
MOCK_ONBOARDING_QUESTIONS_CYCLE: list[dict[str, Any]] = (
    _create_mock_onboarding_cycle_from_flow(ONBOARDING_FLOWS)
)
mock_onboarding_question_index: int = 0
MOCK_ASSESSMENT_QUESTIONS_CONTENT: dict[str, dict[int, str]] = (
    _create_mock_assessment_content_from_flow(ASSESSMENT_FLOWS)
)

# --- LLM Generator, Tools, Pipeline Creation Functions ... ---
# (These remain the same as the last fully refined version you have,
#  including no @cache on get_llm_generator, and corrected
#  ChatPromptBuilder required_variables)

def get_llm_generator() -> OpenAIChatGenerator | None: # Python 3.10+ for | None
    """Initializes and returns an OpenAIChatGenerator instance."""
    # ... (implementation from previous complete script)
    if USE_MOCK_LLM and not os.getenv("OPENAI_API_KEY"):
        logger.info("Mock mode active and no OpenAI key; LLM generator not created.")
        return None
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        llm_generator = OpenAIChatGenerator(api_key=openai_api_key, model=llm_model)
        logger.info("OpenAI Chat Generator instance created for model %s.", llm_model)
        return llm_generator
    except ValueError:
        logger.warning("OPENAI_API_KEY not found. LLM Generator not created (OK if USE_MOCK_LLM=true and pipelines bypass LLM use).")
        return None

def extract_onboarding_data(**kwargs) -> dict[str, Any]:
    """Placeholder function for Haystack Tool 'extract_onboarding_data'."""
    logger.debug("Tool 'extract_onboarding_data' called with: %s", kwargs)
    return kwargs

@cache
def create_onboarding_tool() -> Tool:
    """Creates the Haystack Tool for onboarding data extraction."""
    # ... (Full tool definition from your script)
    tool_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "province": {"type": "string", "description": "The user's province.", "enum": ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga", "Northern Cape", "North West", "Western Cape"]},
            "area_type": {"type": "string", "description": "The type of area the user lives in.", "enum": ["City", "Township or suburb", "Town", "Farm or smallholding", "Village", "Rural area"]},
            # ... (all other properties) ...
            "phone_ownership": {"type": "string", "description": "Whether the user owns their phone.", "enum": ["Yes", "No", "Skip"]},
        },
        "additionalProperties": {"type": "string", "description": "Any other valuable maternal health-related information extracted."},
    }
    return Tool(name="extract_onboarding_data", description="Extract structured data points...", function=extract_onboarding_data, parameters=tool_parameters)

@cache
def create_next_onboarding_question_pipeline() -> Pipeline | None:
    """Creates pipeline for LLM to select and contextualize next question."""
    # ... (Implementation from the previous full script, with corrected required_variables)
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Next Onboarding Q Pipeline.")
        return None
    pipeline = Pipeline()
    prompt_template = """
You are an assistant helping to onboard a new user for a maternal health chatbot.
User Context so far:
{% for key, value in user_context.items() %}- {{ key }}: {{ value if value is not none else 'Not Set' }}
{% endfor %}
Recent Chat History (User/Assistant):
{% for message in chat_history[-6:] %}- {{ message }}
{% endfor %}
Here is a list of remaining questions we can ask to complete their profile.
Each question is identified by a 'question_number' and what data it 'collects'.
Remaining questions:
{% for q in remaining_questions %}
- (Number: {{ q.question_number }}) "{{ q.content }}" (Collects: {{ q.collects }})
{% endfor %}
Based on the User Context and Chat History, and to make the conversation flow naturally,
which single question from the 'Remaining questions' list is the most appropriate to ask next?
Your response MUST be a valid JSON object with two keys:
1. "chosen_question_number": The integer 'question_number' of your chosen question.
2. "contextualized_question": Your rephrased, contextualized version of the chosen question to ask the user.
   Make it conversational. If no prior context, ask directly. Do not change the core intent.

JSON Response:
"""
    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                     required_variables=["user_context", "remaining_questions", "chat_history"])
    json_validator = JsonSchemaValidator(json_schema=NEXT_QUESTION_SCHEMA)
    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator:
        pipeline.add_component("llm", llm_generator)
        pipeline.add_component("json_validator", json_validator)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "json_validator.messages")
    logger.info("Created Next Question Selection Pipeline.")
    return pipeline


@cache
def create_onboarding_data_extraction_pipeline() -> Pipeline | None:
    """Creates pipeline for LLM to extract onboarding data using a tool."""
    # ... (Implementation from the previous full script, with corrected required_variables)
    llm_generator_for_tool_use = get_llm_generator()
    if not llm_generator_for_tool_use and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Onboarding Data Extraction Pipeline.")
        return None
    pipeline = Pipeline()
    prompt_template = """
You are helping extract onboarding data from a user's response...
User's latest message: "{{ user_response }}"
... (rest of your detailed extraction prompt) ...
""" # Shortened for brevity
    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                     required_variables=["user_context", "chat_history", "user_response"])
    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator_for_tool_use:
        extraction_tool = create_onboarding_tool()
        llm_generator_for_tool_use.tools = [extraction_tool]
        pipeline.add_component("llm", llm_generator_for_tool_use)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
    logger.info("Created Onboarding Data Extraction Pipeline.")
    return pipeline

@cache
def create_assessment_contextualization_pipeline() -> Pipeline | None:
    """Creates pipeline to contextualize assessment questions using a retriever."""
    # ... (Implementation from the previous full script, with corrected required_variables)
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Assessment Contextualization Pipeline.")
        return None
    pipeline = Pipeline()
    prompt_template = """
You are an assistant helping to personalize assessment questions...
User Context:
{% for key, value in user_context.items() %}- {{ key }}: {{ value if value is not none else 'Not Set' }}
{% endfor %}
Review assessment question for step {{ documents[0].meta.question_number }}.
Original: {{ documents[0].content }}
...
Contextualized Question:""" # Shortened for brevity
    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                     required_variables=["user_context", "documents"])
    document_store = setup_document_store()
    if not document_store: logger.error("Doc store NA for Assess Ctx Pipeline."); return None
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
def create_assessment_response_validator_pipeline() -> Pipeline | None:
    """Creates pipeline to validate (normalize) assessment responses."""
    # ... (Implementation from the previous full script, with corrected required_variables)
    llm_generator = get_llm_generator()
    if not llm_generator and not USE_MOCK_LLM:
        logger.error("LLM Gen not available & not mock mode. Cannot create Assessment Response Validation Pipeline.")
        return None
    pipeline = Pipeline()
    prompt_template = """
You validate user responses for an assessment...
User Response: {{ user_response }}
Validated Response:""" # Shortened for brevity
    prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template)],
                                     required_variables=["user_response"])
    pipeline.add_component("prompt_builder", prompt_builder)
    if llm_generator:
        pipeline.add_component("llm", llm_generator)
        pipeline.connect("prompt_builder.prompt", "llm.messages")
    logger.info("Created Assessment Response Validation Pipeline.")
    return pipeline


# --- Running Pipelines (These use the dynamically generated MOCK_* variables) ---

def run_next_onboarding_question_pipeline(
    pipeline: Pipeline | None,
    user_context: dict[str, Any],
    remaining_questions: list[dict[str, Any]],
    chat_history: list[str],
) -> dict[str, Any] | None:
    """
    Runs pipeline to select next onboarding question or uses dynamic mocks.
    """
    global mock_onboarding_question_index
    if USE_MOCK_LLM:
        logger.info("MOCKING: run_next_onboarding_question_pipeline")
        if not remaining_questions:
            logger.info("Mocking: No remaining questions.")
            return None
        
        mock_data: dict[str, Any] | None = None
        # Use the dynamically generated MOCK_ONBOARDING_QUESTIONS_CYCLE
        if not MOCK_ONBOARDING_QUESTIONS_CYCLE: # If dynamic creation failed
             logger.warning("Dynamic MOCK_ONBOARDING_QUESTIONS_CYCLE is empty. Using hardcoded fallback mock for next_q.")
             q_def_fallback = remaining_questions[0]
             return {"chosen_question_number": q_def_fallback["question_number"],
                     "contextualized_question": f"Mocked Emergency Fallback: {q_def_fallback['content']}",
                     "collects_field": q_def_fallback.get("collects"), "fallback_used": True}

        for i in range(len(MOCK_ONBOARDING_QUESTIONS_CYCLE)):
            idx = (mock_onboarding_question_index + i) % \
                len(MOCK_ONBOARDING_QUESTIONS_CYCLE)
            potential_mock = MOCK_ONBOARDING_QUESTIONS_CYCLE[idx]
            if any(
                q["question_number"] == potential_mock["chosen_question_number"]
                for q in remaining_questions
            ):
                mock_data = {**potential_mock} # No longer need to add fallback_used if already in mock
                mock_onboarding_question_index = (idx + 1) % \
                    len(MOCK_ONBOARDING_QUESTIONS_CYCLE)
                break
        
        if not mock_data: # Fallback if no cycling mock matches
            q_def = remaining_questions[0]
            mock_data = {
                "chosen_question_number": q_def["question_number"],
                "contextualized_question": (
                    f"Mocked Fallback from remaining: {q_def['content']}"
                ),
                "collects_field": q_def.get("collects"),
                "fallback_used": True,
            }
        logger.info("Mocked next question response: %s", mock_data)
        return mock_data

    # ... (Actual LLM path for run_next_onboarding_question_pipeline remains the same
    #      as the fully corrected version from previous responses, ensuring it returns
    #      the dict with chosen_question_number, contextualized_question,
    #      collects_field, fallback_used) ...
    current_llm_generator = get_llm_generator()
    if not pipeline or not current_llm_generator:
        logger.warning("LLM pipeline/gen not available. Using fallback for next_q.")
        if remaining_questions:
            q_def = remaining_questions[0]
            return {"chosen_question_number": q_def["question_number"], "contextualized_question": q_def["content"], "collects_field": q_def.get("collects"), "fallback_used": True}
        return None
    logger.info("CALLING ACTUAL LLM: run_next_onboarding_question_pipeline")
    try:
        result = pipeline.run(data={"user_context": user_context, "remaining_questions": remaining_questions, "chat_history": chat_history})
        validator_output = result.get("json_validator", {}); validated_list = validator_output.get("validated", [])
        if validated_list and isinstance(validated_list[0], ChatMessage):
            json_str = validated_list[0].content; chosen_data = json.loads(json_str)
            q_num = chosen_data.get("chosen_question_number"); ctx_q = chosen_data.get("contextualized_question")
            coll_f = next((q.get("collects") for q in remaining_questions if q.get("question_number") == q_num), None)
            logger.info("LLM chose Q#: %s, collects: %s", q_num, coll_f)
            return {"chosen_question_number": q_num, "contextualized_question": ctx_q, "collects_field": coll_f, "fallback_used": False}
        else: logger.warning("LLM/Validator failed: %s. Using fallback.", result)
    except Exception as e: logger.error("Error in next_onboarding_q_pipeline: %s", e, exc_info=True)
    if remaining_questions:
        q_def = remaining_questions[0]; logger.warning("Fallback (actual path): Q# %s", q_def["question_number"])
        return {"chosen_question_number": q_def["question_number"], "contextualized_question": q_def["content"], "collects_field": q_def.get("collects"), "fallback_used": True}
    return None

def run_onboarding_data_extraction_pipeline(
    pipeline: Pipeline | None, user_response: str, user_context: dict[str, Any],
    chat_history: list[str], expected_collects_field: str | None = None,
) -> dict[str, Any]:
    """Runs pipeline to extract data from user's onboarding response."""
    if USE_MOCK_LLM:
        logger.info(
            "MOCKING: data_extraction, expecting for field: %s",
            expected_collects_field,
        )
        mock_data: dict[str, Any] = {}
        if expected_collects_field == "province":
            if "gauteng" in user_response.lower():
                mock_data["province"] = "Gauteng"
            elif "western cape" in user_response.lower():
                 mock_data["province"] = "Western Cape"
            elif "north west" in user_response.lower(): # From Naledi scenario
                 mock_data["province"] = "North West"
            else:
                mock_data["province"] = "Mocked KZN" # Default if no match
        elif expected_collects_field == "area_type":
            if "farm" in user_response.lower() or "rural" in user_response.lower():
                mock_data["area_type"] = "Farm or smallholding"
            else:
                mock_data["area_type"] = "Mocked City"
        elif expected_collects_field == "relationship_status":
            if "single" in user_response.lower():
                mock_data["relationship_status"] = "Single"
            else:
                mock_data["relationship_status"] = "Mocked Relationship"
        elif expected_collects_field == "education_level":
            if "grade 9" in user_response.lower() or "some high" in user_response.lower():
                mock_data["education_level"] = "Some high school"
            else:
                mock_data["education_level"] = "Mocked Finished high school"
        elif expected_collects_field == "hunger_days":
            if "3 or 4" in user_response or "3-4" in user_response:
                mock_data["hunger_days"] = "3-4 days"
            else:
                mock_data["hunger_days"] = "Mocked 0 days"
        elif expected_collects_field == "num_children":
            if "one" in user_response.lower() or "1" in user_response:
                mock_data["num_children"] = "1"
            else:
                mock_data["num_children"] = "Mocked 0"
        elif expected_collects_field == "phone_ownership":
            if "no" in user_response.lower() or "share" in user_response.lower():
                mock_data["phone_ownership"] = "No"
            else:
                mock_data["phone_ownership"] = "Mocked Yes"
        else:
            logger.warning(
                "No specific mock for field: '%s'. Generic mock.",
                expected_collects_field
            )
            mock_data[expected_collects_field or "unknown_mock_field"] = (
                f"Mocked value for: '{user_response[:20]}...'"
            )
        if "worried" in user_response.lower():
            mock_data["additional_sentiment"] = "User expressed worry (mocked)"
        logger.info("Mocked extracted data: %s", mock_data)
        return mock_data

    # ... (Actual LLM path for run_onboarding_data_extraction_pipeline remains the same
    #      as the fully corrected version from previous responses) ...
    current_llm_generator = get_llm_generator()
    if not pipeline or not current_llm_generator:
        logger.warning("LLM pipeline/gen for extraction not available. Empty return.")
        return {}
    logger.info("CALLING ACTUAL LLM: data_extraction (for field: %s)", expected_collects_field or "any")
    try:
        result = pipeline.run(data={"user_context": user_context, "chat_history": chat_history, "user_response": user_response})
        llm_output = result.get("llm", {}); replies = llm_output.get("replies", [])
        if replies and isinstance(replies[0], ChatMessage):
            tool_calls = replies[0].tool_calls or []
            if tool_calls and isinstance(tool_calls[0], dict):
                args_str = tool_calls[0].get("arguments", "{}")
                try: args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError: args = {}
                if isinstance(args, dict): logger.info("LLM extracted (tool args): %s", args); return args
        logger.warning("No valid tool calls/replies in extraction.")
        return {}
    except Exception as e:
        logger.error("Error running extraction pipeline: %s", e, exc_info=True); return {}


def run_assessment_contextualization_pipeline(
    pipeline: Pipeline | None, flow_id: str, question_number: int,
    user_context: dict[str, Any],
) -> str | None:
    """Runs pipeline to contextualize an assessment question."""
    if USE_MOCK_LLM:
        logger.info(
            "MOCKING: assessment_contextualization for Q#%s", question_number
        )
        # Use the dynamically generated MOCK_ASSESSMENT_QUESTIONS_CONTENT
        raw_q_from_mock_def = MOCK_ASSESSMENT_QUESTIONS_CONTENT.get(flow_id, {}).get(
            question_number,
            f"Default Dynamic Mock Q{question_number} for {flow_id}"
        )
        # The raw_q_from_mock_def already contains "Mock Assessment QX: content"
        # So we might not need to add "Mocked (QX):" again if it's already descriptive.
        # For this example, let's assume raw_q_from_mock_def is just the plain question from flow.
        # If MOCK_ASSESSMENT_QUESTIONS_CONTENT stores "Mocked: content", then:
        # mock_q = (f"Context for {user_context.get('age', 'N/A')}: "
        #           f"{raw_q_from_mock_def}")
        # If it stores plain content, then this is better:
        mock_q = (
             f"Mocked (Q{question_number}): Considering user age "
             f"{user_context.get('age', 'N/A')}, {raw_q_from_mock_def}"
        )

        logger.info("Mocked contextualized assessment question: %s", mock_q)
        return mock_q

    # ... (Actual LLM path for run_assessment_contextualization_pipeline remains
    #      the same as the fully corrected version from previous responses) ...
    current_llm_generator = get_llm_generator()
    if not pipeline or not current_llm_generator:
        logger.warning("LLM pipeline/gen for assess ctx not available. Fallback.")
        doc_store = setup_document_store()
        if doc_store:
            try:
                retriever = FilterRetriever(document_store=doc_store)
                filters = {"operator": "AND", "conditions": [{"field": "meta.flow_id", "operator": "==", "value": flow_id}, {"field": "meta.question_number", "operator": "==", "value": question_number}]}
                retrieved = retriever.run(filters=filters).get("documents", [])
                if retrieved and isinstance(retrieved[0], Document): return retrieved[0].content
            except Exception as e_fb: logger.error("Fallback retriever failed: %s", e_fb)
        return None
    logger.info("CALLING ACTUAL LLM: assessment_contextualization for Q#%s", question_number)
    try:
        filters = {"operator": "AND", "conditions": [{"field": "meta.flow_id", "operator": "==", "value": flow_id}, {"field": "meta.question_number", "operator": "==", "value": question_number}]}
        result = pipeline.run(data={"retriever": {"filters": filters}, "user_context": user_context})
        llm_output = result.get("llm", {}); replies = llm_output.get("replies", [])
        if replies and isinstance(replies[0], ChatMessage): return replies[0].content
        logger.warning("No valid replies for contextualization.")
        return None
    except Exception as e: logger.error("Error in assess_context_pipeline: %s", e, exc_info=True); return None


def run_assessment_response_validator_pipeline(
    pipeline: Pipeline | None, user_response: str
) -> str | None:
    """Runs pipeline to validate user's assessment response."""
    if USE_MOCK_LLM:
        logger.info("MOCKING: assessment_response_validator")
        processed = f"MockValidated: {user_response.strip().capitalize()}"
        if "nonsense" in user_response.lower() or len(user_response) < 3:
            processed = "nonsense"
        logger.info("Mocked validated response: %s", processed)
        return processed

    # ... (Actual LLM path for run_assessment_response_validator_pipeline
    #      remains the same as the fully corrected version) ...
    current_llm_generator = get_llm_generator()
    if not pipeline or not current_llm_generator:
        logger.warning("LLM pipeline/gen for validation not available. Passthrough.")
        return user_response
    logger.info("CALLING ACTUAL LLM: assessment_response_validator")
    try:
        result = pipeline.run(data={"user_response": user_response})
        llm_output = result.get("llm", {}); replies = llm_output.get("replies", [])
        if replies and isinstance(replies[0], ChatMessage): return replies[0].content
        logger.warning("No valid replies in LLM for validation.")
        return None
    except Exception as e: logger.error("Error in assess_validator_pipeline: %s", e, exc_info=True); return None