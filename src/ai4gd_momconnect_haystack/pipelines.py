import json
import logging
import re
from functools import cache
from typing import Any

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import FilterRetriever
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret

from .doc_store import setup_document_store
from .pipeline_prompts import (
    ANC_SURVEY_CONTEXTUALIZATION_PROMPT,
    ASSESSMENT_CONTEXTUALIZATION_PROMPT,
    ASSESSMENT_END_RESPONSE_VALIDATOR_PROMPT,
    ASSESSMENT_DATA_EXTRACTION_PROMPT,
    ASSESSMENT_RESPONSE_VALIDATOR_PROMPT,
    BEHAVIOUR_DATA_EXTRACTION_PROMPT,
    CLINIC_VISIT_DATA_EXTRACTION_PROMPT,
    FAQ_PROMPT,
    INTENT_DETECTION_PROMPT,
    NEXT_ONBOARDING_QUESTION_PROMPT,
    ONBOARDING_DATA_EXTRACTION_PROMPT,
    REPHRASE_QUESTION_PROMPT,
    SURVEY_DATA_EXTRACTION_PROMPT,
    DATA_UPDATE_PROMPT,
    SUMMARY_CONFIRMATION_INTENT_PROMPT,
)

# --- Configuration ---
logger = logging.getLogger(__name__)

# --- JSON Schemas ---
NEXT_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_question_number": {
            "type": "integer",
            "description": "The question number selected from the remaining questions list",
        },
        "contextualized_question": {
            "type": "string",
            "description": "The contextualized version of the chosen question",
        },
    },
    "required": ["chosen_question_number", "contextualized_question"],
    "additionalProperties": False,
}

ASSESSMENT_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "contextualized_question": {
            "type": "string",
            "description": "The contextualized version of the assessment question, NOT including the list of possible answers.",
        }
    },
    "required": ["contextualized_question"],
    "additionalProperties": False,
}

SURVEY_QUESTION_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "contextualized_question": {
            "type": "string",
            "description": "The contextualized, user-facing version of the survey question.",
        }
    },
    "required": ["contextualized_question"],
    "additionalProperties": False,
}

INTENT_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "description": "The classified intent of the user's message.",
            "enum": [
                "JOURNEY_RESPONSE",
                "QUESTION_ABOUT_STUDY",
                "HEALTH_QUESTION",
                "ASKING_TO_STOP_MESSAGES",
                "ASKING_TO_DELETE_DATA",
                "REPORTING_AIRTIME_NOT_RECEIVED",
                "CHITCHAT",
            ],
        }
    },
    "required": ["intent"],
}

REPHRASED_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "rephrased_question": {
            "type": "string",
            "description": "The rephrased, user-facing version of the question.",
        }
    },
    "required": ["rephrased_question"],
    "additionalProperties": False,
}


# --- The ANC Survey flow, statically defined ---
ANC_SURVEY_FLOW_LOGIC = {
    # --- Entry Point ---
    "intro": lambda ctx: "Q_why_not_go"
    if ctx.get("intro") == "NO"
    else "start_going_soon"
    if ctx.get("intro") == "SOON"
    else "Q_seen",
    # --- Branch 1: Going Soon ---
    "start_going_soon": lambda ctx: None,  # Ends flow, triggers 3-day reminder
    # --- Branch 2: Not Going ---
    "Q_why_not_go": lambda ctx: "intent",
    # --- Branch 3: Went to Clinic ---
    "Q_seen": lambda ctx: "seen_yes" if ctx.get("Q_seen") == "YES" else "Q_seen_no",
    # > Sub-branch: Was Seen
    "seen_yes": lambda ctx: "Q_bp"
    if ctx.get("seen_yes") == "YES"
    else "mom_ANC_remind_me_01",
    "Q_bp": lambda ctx: "Q_experience",
    "Q_experience": lambda ctx: "bad"
    if ctx.get("Q_experience") in ["EXP_BAD", "EXP_VERY_BAD"]
    else "good",
    "good": lambda ctx: "Q_visit_good"
    if ctx.get("good") == "YES"
    else "mom_ANC_remind_me_01",
    "bad": lambda ctx: "Q_visit_bad"
    if ctx.get("bad") == "YES"
    else "mom_ANC_remind_me_01",
    "Q_visit_bad": lambda ctx: "Q_challenges",
    "Q_visit_good": lambda ctx: "Q_challenges",
    "Q_challenges": lambda ctx: "intent",
    # > Sub-branch: Was at clinic but NOT Seen
    "Q_seen_no": lambda ctx: "Q_why_no_visit"
    if ctx.get("Q_seen_no") == "YES"
    else "mom_ANC_remind_me_01",
    "Q_why_no_visit": lambda ctx: "intent",
    # --- Converging Paths & End States ---
    "intent": lambda ctx: (
        "feedback_if_first_survey" if ctx.get("first_survey") else "end"
    )
    if ctx.get("intent") == "YES"
    else "not_going_next_one",
    "not_going_next_one": lambda ctx: "feedback_if_first_survey"
    if ctx.get("first_survey")
    else "end",
    "feedback_if_first_survey": lambda ctx: "end_if_feedback",
    "mom_ANC_remind_me_01": lambda ctx: None,
    "end_if_feedback": lambda ctx: None,
    "end": lambda ctx: None,
}


def get_next_anc_survey_step(current_step: str, user_context: dict) -> str | None:
    """Determines the next step based on the defined ANC survey flow."""
    if current_step not in ANC_SURVEY_FLOW_LOGIC:
        return None

    next_step_func = ANC_SURVEY_FLOW_LOGIC[current_step]
    return next_step_func(user_context)


# --- LLM Generator ---
def get_llm_generator() -> OpenAIChatGenerator | None:
    """
    Returns an instance of OpenAIChatGenerator if the API key is set in the environment.
    """
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_generator = OpenAIChatGenerator(
            api_key=openai_api_key, model="gpt-3.5-turbo"
        )
        logger.info("OpenAI Chat Generator instance created for pipelines.")
    except ValueError:
        logger.error(
            "OPENAI_API_KEY environment variable not found. Cannot create OpenAI Chat Generator."
        )
        return None
    if not llm_generator:
        logger.error("OpenAI Chat Generator not available. Cannot create pipelines.")
        return None
    return llm_generator


# --- Tools ---
def extract_onboarding_data(**kwargs) -> dict[str, Any]:
    """
    Receives extracted data from the LLM tool call via its arguments.
    This function acts as a placeholder; its primary role is to define
    the tool structure for the LLM. It simply returns the arguments it receives.
    """
    logger.info(f"Tool 'extract_onboarding_data' would be called with: {kwargs}")
    return kwargs


def extract_clinic_visit_data(**kwargs) -> dict[str, Any]:
    """
    Acts as a placeholder for the clinic visit survey tool.
    It simply returns the arguments it receives.
    """
    logger.info(f"Tool 'extract_clinic_visit_data' would be called with: {kwargs}")
    return kwargs


def create_onboarding_tool() -> Tool:
    """Creates the data extraction tool for onboarding."""

    tool = Tool(
        name="extract_onboarding_data",
        description="""
        Extract structured data points like province, area_type, relationship_status, etc., from the user's latest message. Crucially, also extract any other information mentioned that seems relevant and valuable in the context of maternal health or the use of a maternal health chatbot (e.g., mentions of specific symptoms, appointments, social support, concerns or preferences), even if not explicitly listed as a parameter.
        """,
        function=extract_onboarding_data,
        parameters={
            "type": "object",
            "properties": {
                "province": {
                    "type": "string",
                    "description": "The user's province.",
                    "enum": [
                        "Eastern Cape",
                        "Free State",
                        "Gauteng",
                        "KwaZulu-Natal",
                        "Limpopo",
                        "Mpumalanga",
                        "Northern Cape",
                        "North West",
                        "Western Cape",
                    ],
                },
                "area_type": {
                    "type": "string",
                    "description": "The type of area the user lives in.",
                    "enum": [
                        "City",
                        "Township or suburb",
                        "Town",
                        "Farm or smallholding",
                        "Village",
                        "Rural area",
                        "Skip",
                    ],
                },
                "relationship_status": {
                    "type": "string",
                    "description": "The user's relationship status.",
                    "enum": ["Single", "Relationship", "Married", "Skip"],
                },
                "education_level": {
                    "type": "string",
                    "description": "The user's highest education level.",
                    "enum": [
                        "No school",
                        "Some primary",
                        "Finished primary",
                        "Some high school",
                        "Finished high school",
                        "More than high school",
                        "Don't know",
                        "Skip",
                    ],
                },
                "hunger_days": {
                    "type": "string",
                    "description": "Number of days in the past 7 days the user didn't have enough to eat.",
                    "enum": ["0 days", "1-2 days", "3-4 days", "5-7 days", "Skip"],
                },
                "num_children": {
                    "type": "string",
                    "description": "The number of children the user has.",
                    "enum": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "More than 3",
                        "Why do you ask?",
                        "Skip",
                    ],
                },
                "phone_ownership": {
                    "type": "string",
                    "description": "Whether the user owns their phone.",
                    "enum": ["Yes", "No", "Skip"],
                },
            },
            "additionalProperties": {
                "type": "string",
                "description": "Any other valuable maternal health-related information extracted.",
            },
        },
    )

    return tool


def extract_updated_data(**kwargs) -> dict[str, Any]:
    """
    Receives extracted data from the LLM tool call during a data update request.
    This function acts as a placeholder; its primary role is to define
    the tool structure for the LLM. It simply returns the arguments it receives.
    """
    logger.info(f"Tool 'extract_updated_data' would be called with: {kwargs}")
    return kwargs


def create_data_update_tool() -> Tool:
    """Creates the data extraction tool for updating user profile information."""
    # This tool is intentionally similar to the onboarding tool for consistency.
    tool = Tool(
        name="extract_updated_data",
        description="Extracts one or more pieces of user profile information from the user's message and returns them in a structured format.",
        function=extract_updated_data,
        parameters={
            "type": "object",
            "properties": {
                "province": {
                    "type": "string",
                    "description": "The user's province.",
                    "enum": [
                        "Eastern Cape",
                        "Free State",
                        "Gauteng",
                        "KwaZulu-Natal",
                        "Limpopo",
                        "Mpumalanga",
                        "Northern Cape",
                        "North West",
                        "Western Cape",
                    ],
                },
                "area_type": {
                    "type": "string",
                    "description": "The type of area the user lives in.",
                    "enum": [
                        "City",
                        "Township or suburb",
                        "Town",
                        "Farm or smallholding",
                        "Village",
                        "Rural area",
                    ],
                },
                "relationship_status": {
                    "type": "string",
                    "description": "The user's relationship status.",
                    "enum": ["Single", "Relationship", "Married"],
                },
                "education_level": {
                    "type": "string",
                    "description": "The user's highest education level.",
                    "enum": [
                        "No school",
                        "Some primary",
                        "Finished primary",
                        "Some high school",
                        "Finished high school",
                        "More than high school",
                        "Don't know",
                    ],
                },
                "hunger_days": {
                    "type": "string",
                    "description": "Number of days in the past 7 days the user didn't have enough to eat.",
                    "enum": ["0 days", "1-2 days", "3-4 days", "5-7 days"],
                },
                "num_children": {
                    "type": "string",
                    "description": "The number of children the user has.",
                    "enum": ["0", "1", "2", "3", "More than 3"],
                },
                "phone_ownership": {
                    "type": "string",
                    "description": "Whether the user owns their phone.",
                    "enum": ["Yes", "No"],
                },
            },
        },
    )
    return tool


# --- Creation of Pipelines ---
@cache
def create_next_onboarding_question_pipeline() -> Pipeline:
    """
    Creates a pipeline where an LLM selects the best next onboarding question
    from a list of remaining questions, given the user's current context.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Onboarding Data Extraction Pipeline."
        )
        return None
    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component(
        "json_validator", JsonSchemaValidator(json_schema=NEXT_QUESTION_SCHEMA)
    )

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created Next Question Selection Pipeline with JSON schema validation.")
    return pipeline


@cache
def create_onboarding_data_extraction_pipeline() -> Pipeline | None:
    """
    Creates a pipeline using tools to extract structured data from user responses.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Onboarding Data Extraction Pipeline."
        )
        return None

    extraction_tool = create_onboarding_tool()
    llm_generator.tools = [extraction_tool]

    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Onboarding Data Extraction Pipeline with Tools.")
    return pipeline


def create_assessment_contextualization_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to fetch an assessment question based on flow_id and question_number,
    then contextualize it slightly using an LLM based on user context.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment Contextualization Pipeline."
        )
        return None
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system("""
{# This is a placeholder template to declare input variables to be overridden at runtime#}
{% for doc in documents %}{{ doc.content }}{% endfor %}
{{ user_context }}
""")
        ],
        required_variables=["documents", "user_context"],
    )

    document_store = setup_document_store()
    retriever = FilterRetriever(document_store=document_store)
    json_validator = JsonSchemaValidator(json_schema=ASSESSMENT_CONTEXT_SCHEMA)

    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created Assessment Contextualization Pipeline.")
    return pipeline


@cache
def create_assessment_response_validator_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to validate the user's response to an assessment question
    against a dynamic list of valid responses, with a guaranteed structured output.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment Response Validation Pipeline."
        )
        return None
    pipeline = Pipeline()

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created Assessment Response Validation Pipeline with JSON Schema validation."
    )
    return pipeline


def create_assessment_data_extraction_pipeline() -> Pipeline | None:
    """

    Creates a pipeline to extract structured data for an assessment.
    NOTE: This is a copy of create_clinic_visit_data_extraction_pipeline.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment Data Extraction Pipeline."
        )
        return None

    pipeline = Pipeline()

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created Assessment Data Extraction Pipeline with JSON Schema validation."
    )
    return pipeline


@cache
def create_behaviour_data_extraction_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to extract structured data from a user's
    response during the KAB Behaviour assessment.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Behaviour Data Extraction Pipeline."
        )
        return None

    pipeline = Pipeline()

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created KAB Behaviour Data Extraction Pipeline with JSON Schema validation."
    )
    return pipeline


@cache
def create_assessment_end_response_validator_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to validate the user's response to an end-of-assessment
    message against a dynamic list of valid responses.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment End Response Validation Pipeline."
        )
        return None
    pipeline = Pipeline()

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created Assessment End Response Validation Pipeline with JSON Schema validation."
    )
    return pipeline


@cache
def create_anc_survey_contextualization_pipeline() -> Pipeline | None:
    """
    Creates a new pipeline to contextualize a specific ANC survey question
    based on the user context and chat history.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator not available. Cannot create ANC Survey Contextualization Pipeline."
        )
        return None

    pipeline = Pipeline()

    json_validator = JsonSchemaValidator(json_schema=SURVEY_QUESTION_CONTEXT_SCHEMA)

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created ANC Survey Contextualization Pipeline.")
    return pipeline


def create_clinic_visit_data_extraction_pipeline() -> Pipeline | None:
    """
    Creates a pipeline using a tool to extract structured data from a user's
    response during the ANC survey.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create ANC Survey Data Extraction Pipeline."
        )
        return None

    pipeline = Pipeline()

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created dynamic ANC Survey Data Extraction Pipeline with JSON Schema validation."
    )
    return pipeline


@cache
def create_intent_detection_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to classify the user's intent.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Intent Detection Pipeline."
        )
        return None
    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Intent Detection Pipeline with JSON Schema validation.")
    return pipeline


@cache
def create_faq_answering_pipeline() -> Pipeline | None:
    """
    Creates a RAG pipeline to answer user questions using the FAQ documents.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create FAQ Answering Pipeline."
        )
        return None
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system("""
{# This is a placeholder template to declare input variables to be overridden at runtime#}
{% for doc in documents %}{{ doc.content }}{% endfor %}
{{ user_question }}
""")
        ],
        required_variables=["documents", "user_question"],
    )

    document_store = setup_document_store()
    retriever = FilterRetriever(document_store=document_store)

    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created FAQ Answering Pipeline.")
    return pipeline


@cache
def create_rephrase_question_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to rephrase a question when a user's response is unclear.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator not available. Cannot create Rephrase Question Pipeline."
        )
        return None

    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Rephrase Question Pipeline.")
    return pipeline


@cache
def create_survey_data_extraction_pipeline() -> Pipeline | None:
    """
    Creates a pipeline for survey data extraction that returns a structured
    object with confidence and match type.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Survey Data Extraction Pipeline."
        )
        return None

    pipeline = Pipeline()
    # Define the schema for the richer output
    json_schema = {
        "type": "object",
        "properties": {
            "validated_response": {"type": "string"},
            "match_type": {"type": "string", "enum": ["exact", "inferred", "no_match"]},
            "confidence": {"type": "string", "enum": ["high", "low"]},
        },
        "required": ["validated_response", "match_type", "confidence"],
    }
    json_validator = JsonSchemaValidator(json_schema=json_schema)

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created Survey Data Extraction Pipeline with confidence scoring.")
    return pipeline


@cache
def create_data_update_pipeline() -> Pipeline | None:
    """
    Creates a pipeline that uses a tool to extract updated profile data
    from a user's free-text request.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Data Update Pipeline."
        )
        return None

    update_tool = create_data_update_tool()
    llm_generator.tools = [update_tool]

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Data Update Pipeline with Tools.")
    return pipeline


# --- Running Pipelines ---
def run_next_onboarding_question_pipeline(
    user_context: dict[str, Any],
    remaining_questions: list[dict],
    is_first_question=False,
    is_last_question=False,
) -> dict[str, Any] | None:
    """
    Run the next onboarding question selection pipeline and return the chosen question.

    Args:
        user_context: Previously collected user data.
        remaining_questions: List of remaining questions to choose from.

    Returns:
        Dictionary containing the chosen question number and contextualized question.
    """
    pipeline = create_next_onboarding_question_pipeline()
    if not pipeline:
        logger.warning(
            "Failed to create Next Onboarding Question pipeline. Using fallback."
        )
    else:
        try:
            chat_template = [ChatMessage.from_system(NEXT_ONBOARDING_QUESTION_PROMPT)]
            result = pipeline.run(
                {
                    "prompt_builder": {
                        "template": chat_template,
                        "template_variables": {
                            "user_context": user_context,
                            "remaining_questions": remaining_questions,
                            "is_first_question": is_first_question,
                            "is_last_question": is_last_question,
                        },
                    }
                }
            )

            validated_message = result["json_validator"]["validated"][0]
            chosen_data = json.loads(validated_message.text)
            chosen_question_number = chosen_data["chosen_question_number"]
            contextualized_question = chosen_data["contextualized_question"]

            logger.info(
                f"LLM chose question with question_number: {chosen_question_number}"
            )
            logger.info(f"LLM contextualized question: {contextualized_question}")

            return {
                "chosen_question_number": chosen_question_number,
                "contextualized_question": contextualized_question,
            }
        except (KeyError, IndexError):
            logger.warning(
                "LLM response for question selection was invalid or missing. Using fallback. Result: %s",
                result,
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "LLM failed to produce valid JSON in question selection pipeline: %s. Using fallback.",
                e,
            )
        except Exception as e:
            logger.warning(
                "Unexpected error in next onboarding question pipeline execution: %s", e
            )

    # Fallback logic
    if remaining_questions:
        chosen_question = remaining_questions[0]
        logger.warning(
            "Falling back to first remaining question: question_number %s",
            chosen_question["question_number"],
        )
        return {
            "chosen_question_number": chosen_question["question_number"],
            "contextualized_question": chosen_question["content"],
        }

    logger.error("No remaining questions available to fall back on.")
    return None


def run_onboarding_data_extraction_pipeline(
    user_response: str,
    user_context: dict[str, Any],
    current_question: str,
) -> dict[str, Any]:
    """
    Run the onboarding data extraction pipeline and return extracted data.

    Args:
        user_response: User's latest message.
        user_context: Previously collected user data.

    Returns:
        Dictionary containing extracted data points.
    """
    pipeline = create_onboarding_data_extraction_pipeline()
    if not pipeline:
        logger.warning("Failed to create Onboarding Data Extraction pipeline.")
        return {}

    try:
        chat_template = [ChatMessage.from_system(ONBOARDING_DATA_EXTRACTION_PROMPT)]
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_response": user_response,
                        "user_context": user_context,
                        "current_question": current_question,
                    },
                }
            }
        )

        tool_calls = result["llm"]["replies"][0].tool_calls
        if tool_calls:
            # Assuming the first tool call is the one we want.
            arguments = tool_calls[0].arguments
            if isinstance(arguments, dict):
                logger.info(f"Extracted data: {arguments}")
                return arguments
            else:
                logger.warning(f"Tool arguments are not a dictionary: {arguments}")
        else:
            logger.warning("No tool calls found in LLM response for data extraction.")

    except (KeyError, IndexError, AttributeError) as e:
        logger.warning(
            "Failed to parse LLM response for data extraction: %s. Result structure might be invalid. Result: %s",
            e,
            result,
        )
    except Exception as e:
        logger.error("Error running extraction pipeline: %s", e)

    return {}


def run_assessment_contextualization_pipeline(
    flow_id: str, question_number: int, user_context: dict[str, Any]
) -> str | None:
    """
    Run the assessment contextualization pipeline to get a contextualized question.

    Args:
        flow_id: The ID of the assessment flow
        question_number: The question number to contextualize
        user_context: Previously collected user data

    Returns:
        Contextualized question string or None if an error occurs
    """
    pipeline = create_assessment_contextualization_pipeline()
    if not pipeline:
        logger.warning("Failed to create Assessment Contextualization pipeline.")
        return None

    chat_template = [ChatMessage.from_system(ASSESSMENT_CONTEXTUALIZATION_PROMPT)]
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.flow_id", "operator": "==", "value": flow_id},
            {
                "field": "meta.question_number",
                "operator": "==",
                "value": question_number,
            },
        ],
    }

    try:
        result = pipeline.run(
            {
                "retriever": {"filters": filters},
                "prompt_builder": {
                    "template": chat_template,
                    "user_context": user_context,
                },
            },
            include_outputs_from=["retriever"],
        )

        retrieved_docs = result.get("retriever", {}).get("documents", [])
        if not retrieved_docs:
            logger.error(
                f"Could not retrieve document for flow '{flow_id}' and question {question_number}."
            )
            return None

        validated_message = result["json_validator"]["validated"][0]
        question_data = json.loads(validated_message.text)
        contextualized_question = question_data["contextualized_question"]

        return contextualized_question.strip()

    except (KeyError, IndexError):
        logger.warning(
            "LLM response for assessment contextualization was invalid or missing. Result: %s",
            result,
        )
    except json.JSONDecodeError as e:
        logger.warning(
            "LLM failed to produce valid JSON in assessment contextualization pipeline: %s.",
            e,
        )
    except Exception as e:
        logger.error(
            "Unexpected error in assessment_contextualization_pipeline execution: %s", e
        )

    return None


def run_assessment_data_extraction_pipeline(
    user_response: str,
    previous_message: str,
    valid_responses: list[str],
) -> str | None:
    """
    Run the assessment data extraction pipeline.
    """
    pipeline = create_assessment_data_extraction_pipeline()
    if not pipeline:
        logger.warning("Failed to create Assessment Data Extraction pipeline.")
        return None

    chat_template = [ChatMessage.from_system(ASSESSMENT_DATA_EXTRACTION_PROMPT)]
    valid_responses_schema = {
        "type": "object",
        "properties": {
            "validated_response": {
                "type": "string",
                "enum": valid_responses + ["nonsense"],
            }
        },
        "required": ["validated_response"],
    }

    result = {}  # Initialize result to an empty dict
    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_response": user_response,
                        "previous_service_message": previous_message,
                    },
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        validated_json = json.loads(validated_message.text)
        validated_response = validated_json["validated_response"]

        if validated_response != "nonsense":
            return validated_response
        else:
            logger.warning(
                "User response '%s' was validated as 'nonsense'.", user_response
            )
            return None
    except (KeyError, IndexError) as e:
        logger.warning(
            f"Validation failed due to missing key: {e}. The raw pipeline result was: {result}"
        )
    except json.JSONDecodeError as e:
        llm_reply = result.get("llm", {}).get("replies", [None])[0]
        logger.warning(
            f"Validation failed due to invalid JSON: {e}. The raw LLM reply was: {llm_reply}"
        )
    except Exception as e:
        logger.warning(
            f"An unexpected error occurred in the pipeline: {e}. The raw pipeline result was: {result}"
        )

    return None


def run_assessment_response_validator_pipeline(
    user_response: str,
    valid_responses: list[str],
    valid_responses_for_prompt: str,
) -> str | None:
    """
    Run the assessment response validator pipeline to validate a user's response.

    Args:
        user_response: User's response to validate.
        valid_responses: A list of valid string responses for the current question.
        valid_responses_for_prompt: A list of valid string responses for the
            current question as the pipeline should see them.

    Returns:
        The validated response string, or None if the response is invalid/nonsense.
    """
    pipeline = create_assessment_response_validator_pipeline()
    if not pipeline:
        logger.warning("Failed to create Assessment Response Validator pipeline.")
        return None

    try:
        valid_responses_schema = {
            "type": "object",
            "properties": {
                "validated_response": {
                    "type": "string",
                    "description": "The validated response from the user.",
                    "enum": valid_responses + ["nonsense"],
                }
            },
            "required": ["validated_response"],
        }

        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": [
                        ChatMessage.from_system(ASSESSMENT_RESPONSE_VALIDATOR_PROMPT)
                    ],
                    "template_variables": {
                        "user_response": user_response,
                        "valid_responses_for_prompt": valid_responses_for_prompt,
                    },
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        validated_json = json.loads(validated_message.text)
        validated_response = validated_json["validated_response"]

        if validated_response != "nonsense":
            return validated_response
        else:
            logger.warning(
                "User response '%s' was validated as 'nonsense'.", user_response
            )
            return None

    except (KeyError, IndexError):
        logger.warning(
            "Validation failed: LLM output did not match schema for user response: '%s'. Result: %s",
            user_response,
            result,
        )
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to decode JSON from LLM response in validation pipeline: %s", e
        )
    except Exception as e:
        logger.error("Error running assessment response validation pipeline: %s", e)

    return None


def run_behaviour_data_extraction_pipeline(
    user_response: str,
    previous_service_message: str,
    valid_responses: list[str],
) -> str | None:
    """
    Run the KAB Behaviour data extraction pipeline and return extracted data.
    """
    pipeline = create_behaviour_data_extraction_pipeline()
    if not pipeline:
        logger.warning("Failed to create Behaviour Data Extraction pipeline.")
        return None

    chat_template = [ChatMessage.from_system(BEHAVIOUR_DATA_EXTRACTION_PROMPT)]
    valid_responses_schema = {
        "type": "object",
        "properties": {
            "validated_response": {
                "type": "string",
                "description": "The validated response from the user.",
                "enum": valid_responses + ["nonsense"],
            }
        },
        "required": ["validated_response"],
    }

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_response": user_response,
                        "previous_service_message": previous_service_message,
                        "valid_responses": valid_responses,
                    },
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        validated_json = json.loads(validated_message.text)
        validated_response = validated_json["validated_response"]

        if validated_response != "nonsense":
            return validated_response
        else:
            logger.warning(
                "User response '%s' was validated as 'nonsense'.", user_response
            )
            return None

    except (KeyError, IndexError):
        logger.warning(
            "Behaviour data extraction failed. LLM output did not match schema for user response: '%s'. Result: %s",
            user_response,
            result,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON from LLM response: {e}")
    except Exception as e:
        logger.warning(f"Error running behaviour data extraction pipeline: {e}")

    return None


def run_assessment_end_response_validator_pipeline(
    user_response: str,
    valid_responses: list[str],
    previous_message: str,
) -> str | None:
    """
    Run the assessment end response validator pipeline to validate a user's response.

    Args:
        user_response: User's response to validate.
        valid_responses: A list of valid string responses for the message.
        previous_message: The message to which the user is responding.

    Returns:
        The validated response string, or None if the response is invalid/nonsense.
    """
    pipeline = create_assessment_end_response_validator_pipeline()
    if not pipeline:
        logger.warning("Failed to create Assessment End Response Validator pipeline.")
        return None

    chat_template = [ChatMessage.from_system(ASSESSMENT_END_RESPONSE_VALIDATOR_PROMPT)]
    valid_responses_schema = {
        "type": "object",
        "properties": {
            "validated_response": {
                "type": "string",
                "description": "The validated response from the user.",
                "enum": valid_responses + ["nonsense"],
            }
        },
        "required": ["validated_response"],
    }

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_response": user_response,
                        "previous_message": previous_message,
                    },
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        validated_json = json.loads(validated_message.text)
        validated_response = validated_json["validated_response"]

        if validated_response != "nonsense":
            return validated_response
        else:
            logger.warning(
                "User response '%s' was validated as 'nonsense'.", user_response
            )
            return None

    except (KeyError, IndexError):
        logger.warning(
            "Validation failed: LLM output did not match schema for user response: '%s'. Result: %s",
            user_response,
            result,
        )
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to decode JSON from LLM response in validation pipeline: %s", e
        )
    except Exception as e:
        logger.warning(
            f"Error running assessment end response validation pipeline: {e}"
        )

    return None


def run_anc_survey_contextualization_pipeline(
    user_context: dict[str, Any],
    chat_history: list[ChatMessage],
    original_question: str,
    valid_responses: list[str],
) -> str:
    """
    Runs the ANC survey contextualization pipeline.

    Args:
        user_context: Previously collected user data.
        chat_history: List of chat history messages.
        original_question: The original, non-contextualized question to be used as a fallback.
        valid_responses: A list of valid string responses for the current question.

    Returns:
        The contextualized question, or the original_question if contextualization fails.
    """
    pipeline = create_anc_survey_contextualization_pipeline()
    if not pipeline:
        logger.warning(
            "Failed to create ANC Survey Contextualization pipeline. Falling back to original question."
        )
        return original_question

    chat_template = chat_history + [
        ChatMessage.from_system(ANC_SURVEY_CONTEXTUALIZATION_PROMPT)
    ]

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_context": user_context,
                        "chat_history": chat_history,
                        "original_question": original_question,
                        "valid_responses": valid_responses,
                    },
                },
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        question_data = json.loads(validated_message.text)
        contextualized_question = question_data.get("contextualized_question")

        if contextualized_question:
            return contextualized_question
        else:
            logger.warning(
                "Contextualization succeeded but 'contextualized_question' key is missing. Falling back."
            )
            return original_question

    except (KeyError, IndexError):
        logger.warning(
            "Contextualization pipeline failed to produce a valid structured response. Falling back to original question."
        )
    except json.JSONDecodeError as e:
        logger.warning(
            "Contextualization pipeline failed to produce valid JSON: %s. Falling back to original question.",
            e,
        )
    except Exception as e:
        logger.warning(f"Error running ANC survey contextualization pipeline: {e}")

    return original_question


def run_clinic_visit_data_extraction_pipeline(
    user_response: str,
    previous_service_message: str,
    valid_responses: list[str],
) -> str | None:
    """
    Run the ANC survey data extraction pipeline and return extracted data.

    Args:
        user_response: User's latest message
        previous_service_message: Previous message sent to the user, to which they are responding.
        valid_responses: A list of valid responses associated with the previous_service_message.

    Returns:
        String containing extracted data point
    """
    pipeline = create_clinic_visit_data_extraction_pipeline()
    if not pipeline:
        logger.warning("Failed to create Clinic Visit Data Extraction pipeline.")
        return None

    chat_template = [ChatMessage.from_system(CLINIC_VISIT_DATA_EXTRACTION_PROMPT)]
    valid_responses_schema = {
        "type": "object",
        "properties": {
            "validated_response": {
                "type": "string",
                "description": "The validated response from the user.",
                "enum": valid_responses + ["nonsense"],
            }
        },
        "required": ["validated_response"],
    }

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_response": user_response,
                        "previous_service_message": previous_service_message,
                    },
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        validated_message = result["json_validator"]["validated"][0]
        validated_json = json.loads(validated_message.text)
        validated_response = validated_json["validated_response"]

        if validated_response != "nonsense":
            return validated_response
        else:
            logger.warning(
                "User response '%s' was validated as 'nonsense'.", user_response
            )
            return None

    except (KeyError, IndexError):
        logger.warning(
            "Clinic visit data extraction failed. LLM output did not match schema for user response: '%s'. Result: %s",
            user_response,
            result,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to decode JSON from LLM response: {e}")
    except Exception as e:
        logger.warning(f"Error running clinic visit data extraction pipeline: {e}")

    return None


def run_intent_detection_pipeline(
    last_question: str, user_response: str, valid_responses: list[str] | None = None
) -> dict[str, Any] | None:
    """
    Runs the intent detection pipeline and safely parses the JSON from the LLM response.

    Args:
        last_question: Previous message sent to the user, to which they are responding.
        user_response: User's latest message
        valid_responses: Optional list of valid responses for the current question.

    Returns:
        Dictionary containing the classified intent.
    """
    pipeline = create_intent_detection_pipeline()
    if not pipeline:
        logger.warning("Failed to create Intent Detection pipeline.")
        return None

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": [ChatMessage.from_system(INTENT_DETECTION_PROMPT)],
                    "template_variables": {
                        "last_question": last_question,
                        "user_response": user_response,
                        "valid_responses": valid_responses or [],
                    },
                },
            }
        )
        llm_reply = result["llm"]["replies"][0].text

        # Use regex to find the content between <json> and </json> tags
        match = re.search(r"<json>(.*?)</json>", llm_reply, re.DOTALL | re.IGNORECASE)

        if match:
            # .group(1) captures the content *inside* the tags
            json_string = match.group(1).strip()
            intent_data = json.loads(json_string)
            logger.info(f"LLM classified intent as: {intent_data}")
            return intent_data
        else:
            logger.warning("Could not find <json> tags in the LLM's reply.")

    except (KeyError, IndexError):
        logger.warning(
            "Intent detection pipeline failed to produce a valid structured response. Result: %s",
            result,
        )
    except json.JSONDecodeError as e:
        logger.warning("Intent pipeline failed to produce valid JSON response: %s", e)
    except Exception as e:
        logger.warning("Error running intent detection pipeline: %s", e)

    return None


def run_faq_pipeline(user_question: str) -> dict[str, Any] | None:
    """
    Runs the FAQ answering pipeline.

    Args:
        user_question: User's question that should be answered.

    Returns:
        Dictionary containing the answer to the user's question, if one was found.
    """
    pipeline = create_faq_answering_pipeline()
    if not pipeline:
        logger.warning("Failed to create FAQ pipeline.")
        return None

    try:
        result = pipeline.run(
            {
                "retriever": {
                    "filters": {
                        "field": "meta.flow_id",
                        "operator": "==",
                        "value": "faqs",
                    }
                },
                "prompt_builder": {
                    "template": [ChatMessage.from_system(FAQ_PROMPT)],
                    "user_question": user_question,
                },
            }
        )
        answer = result["llm"]["replies"][0].text
        return {"answer": answer}

    except (KeyError, IndexError):
        logger.warning(
            "FAQ pipeline failed to produce a valid response. Result: %s", result
        )
    except Exception as e:
        logger.warning("Error running FAQ pipeline: %s", e)

    return None


def run_rephrase_question_pipeline(
    previous_question: str, invalid_input: str, valid_responses: list[str]
) -> str | None:
    """
    Runs the question rephrasing pipeline.

    Args:
        previous_question: The question that the user found confusing.
        invalid_input: The user's confusing response.
        valid_responses: The list of acceptable answer strings.

    Returns:
        The rephrased question string, or None if an error occurs.
    """
    pipeline = create_rephrase_question_pipeline()
    if not pipeline:
        logger.warning("Failed to create Rephrase Question pipeline.")
        return None

    chat_template = [ChatMessage.from_system(REPHRASE_QUESTION_PROMPT)]

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "previous_question": previous_question,
                        "invalid_input": invalid_input,
                        "valid_responses": valid_responses,
                    },
                },
            }
        )

        rephrased_question = result["llm"]["replies"][0].text
        rephrased_question = rephrased_question.strip() if rephrased_question else None

        if rephrased_question:
            return rephrased_question
        else:
            logger.warning(
                "Rephrasing pipeline succeeded but 'rephrased_question' key is missing."
            )

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("Rephrasing pipeline failed to produce a valid response: %s", e)
    except Exception as e:
        logger.error("Unexpected error in rephrase_question_pipeline execution: %s", e)

    return None


def run_survey_data_extraction_pipeline(
    user_response: str,
    previous_service_message: str,
    response_key_map: dict[str, str],
) -> dict | None:
    """
    Runs the confidence-based survey data extraction pipeline.
    """
    pipeline = create_survey_data_extraction_pipeline()
    if not pipeline:
        return None

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": [
                        ChatMessage.from_system(SURVEY_DATA_EXTRACTION_PROMPT)
                    ],
                    "template_variables": {
                        "user_response": user_response,
                        "previous_service_message": previous_service_message,
                        "response_key_map": response_key_map,
                    },
                }
            }
        )
        validated_message = result["json_validator"]["validated"][0]
        return json.loads(validated_message.text)
    except Exception as e:
        logger.error(f"Error running survey data extraction pipeline: {e}")
        return None


def run_data_update_pipeline(user_input: str, user_context: dict) -> dict:
    """
    Runs the pipeline to extract updated data fields from a user's free-text request.
    """
    pipeline = create_data_update_pipeline()
    if not pipeline:
        logger.warning("Failed to create Data Update pipeline.")
        return {}

    try:
        chat_template = [ChatMessage.from_system(DATA_UPDATE_PROMPT)]
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": chat_template,
                    "template_variables": {
                        "user_input": user_input,
                        "user_context": user_context,
                    },
                }
            }
        )

        tool_calls = result["llm"]["replies"][0].tool_calls
        if tool_calls:
            arguments = tool_calls[0].arguments
            if isinstance(arguments, dict):
                logger.info(f"Extracted updated data: {arguments}")
                return arguments
        else:
            logger.info("Data update pipeline ran, but no updates were extracted.")

    except (KeyError, IndexError, AttributeError) as e:
        logger.warning(
            "Failed to parse LLM response for data update: %s. Result: %s",
            e,
            result,
        )
    return {}


# Add these functions anywhere in the file

@cache
def create_summary_intent_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to classify the user's intent at the summary confirmation step.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        return None

    pipeline = Pipeline()
    json_schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["CONFIRM", "UPDATE", "AMBIGUOUS"]},
        },
        "required": ["intent"],
    }
    json_validator = JsonSchemaValidator(json_schema=json_schema)

    pipeline.add_component("prompt_builder", ChatPromptBuilder())
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")
    return pipeline


def run_summary_intent_pipeline(user_input: str) -> str | None:
    """
    Runs the summary confirmation intent classification pipeline.
    """
    pipeline = create_summary_intent_pipeline()
    if not pipeline:
        return "AMBIGUOUS"

    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "template": [ChatMessage.from_system(SUMMARY_CONFIRMATION_INTENT_PROMPT)],
                    "template_variables": {"user_input": user_input},
                }
            }
        )
        validated_message = result["json_validator"]["validated"][0]
        return json.loads(validated_message.text).get("intent")
    except Exception:
        return "AMBIGUOUS"
