import logging
import json
from functools import cache

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import FilterRetriever
from haystack.components.validators import JsonSchemaValidator
from haystack.tools import Tool
from haystack.utils import Secret

from .doc_store import setup_document_store

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
def extract_onboarding_data(**kwargs) -> dict[str, any]:
    """
    Receives extracted data from the LLM tool call via its arguments.
    This function acts as a placeholder; its primary role is to define
    the tool structure for the LLM. It simply returns the arguments it receives.
    """
    logger.info(f"Tool 'extract_onboarding_data' would be called with: {kwargs}")
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
                    "enum": ["0 days", "1-2 days", "3-4 days", "5-7 days"],
                },
                "num_children": {
                    "type": "string",
                    "description": "The number of children the user has.",
                    "enum": ["0", "1", "2", "3", "More than 3", "Why do you ask?"],
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


# --- Creation of Pipelines ---
@cache
def create_next_onboarding_question_pipeline() -> Pipeline | None:
    """
    Creates a pipeline where an LLM selects the best next onboarding question
    from a list of remaining questions, given the user's current context.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Next Onboarding Question Pipeline."
        )
        return None
    pipeline = Pipeline()

    prompt_template = """
    You are an assistant helping to onboard a new user.
    The user has already provided the following information:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Chat History:
    {% for message in chat_history %}
    - {{ message }}
    {% endfor %}

    Here are the remaining questions we can ask to complete their profile:
    {% for q in remaining_questions %}
    - Question: "{{ q.content }}" (collects data for: "{{ q.collects }}", possible correctly formatted values: "{{ q.valid_responses }}", current question_number for reference: {{ q.question_number }})
    {% endfor %}

    Considering the information already collected, the chat history, and the remaining questions,
    which single question would be the most natural and effective to ask next?
    Your goal is to make the onboarding conversational and logical such that a subsequent prompt can be made to try to extract the expected data from the user's response.

    You MUST respond with a valid JSON object containing exactly these fields:
    - "chosen_question_number" (integer): The question_number of the chosen question from the list above
    - "contextualized_question" (string): A contextualized version of the question

    You can reference the existing User Context and Chat History to modify the tonality and/or phrasing,
    or even add emojis, but DO NOT change the core meaning of the question nor introduce ambiguity. Ensure that the chat flows
    smoothly (e.g. the first message in a chat must not start as if there were preceding messages).

    JSON Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "chat_history", "remaining_questions"],
    )

    json_validator = JsonSchemaValidator(json_schema=NEXT_QUESTION_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

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
        required_variables=["user_context", "chat_history", "user_response"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Onboarding Data Extraction Pipeline with Tools.")
    return pipeline


@cache
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

    prompt_template = """
    You are an assistant helping to personalize assessment questions on a maternal health chatbot service.
    The user has already provided the following information:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Review the following assessment question intended for sequence step {{ documents[0].meta.question_number }}.
    If you think it's needed, make minor adjustments to ensure that the question is clear and directly applicable
    to the user's context.
    **Crucially, do not change the core meaning, difficulty, or the scale/format of the question.**
    Just ensure clarity and relevance. If no changes are needed, return the original question.

    Make sure that the list of valid responses is at the end of the contextualized question.

    Original Assessment Question:
    {{ documents[0].content }}
    Valid Responses:
    1 - Not at all confident
    2 - A little confident
    3 - Somewhat confident
    4 - Confident
    5 - Very confident

    Contextualized Question:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "documents"],
    )

    document_store = setup_document_store()
    retriever = FilterRetriever(document_store=document_store)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Assessment Contextualization Pipeline.")
    return pipeline


@cache
def create_assessment_response_validator_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to validate the user's response to an assessment question.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment Response Validation Pipeline."
        )
        return None
    pipeline = Pipeline()

    prompt_template = """
    You are an assistant helping to validate user responses to an assessment question on a maternal health chatbot service.
    The valid possible responses are:
    1 - Not at all confident
    2 - A little confident
    3 - Somewhat confident
    4 - Confident
    5 - Very confident

    User responses are unpredictable. They might reply with a number, or with text, or with both, corresponding to a valid response.
    Or, they might respond with nonsense or gibberish.

    If you think that the user response maps to one of the valid responses, your output must be the text corresponding to that valid response (i.e. without the number).
    If you think that the user response is nonsense or gibberish instead, return "nonsense".

    User Response:
    {{ user_response }}

    Validated Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_response"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Assessment Response Validation Pipeline.")
    return pipeline


# --- Running Pipelines ---
def run_next_onboarding_question_pipeline(
    pipeline: Pipeline,
    user_context: dict[str, any],
    remaining_questions: list[dict],
    chat_history: list[str],
) -> dict[str, any] | None:
    """
    Run the next onboarding question selection pipeline and return the chosen question.

    Args:
        pipeline: The configured pipeline
        user_context: Previously collected user data
        remaining_questions: List of remaining questions to choose from
        chat_history: List of chat history messages

    Returns:
        Dictionary containing the chosen question number and contextualized question
    """
    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "user_context": user_context,
                    "remaining_questions": remaining_questions,
                    "chat_history": chat_history,
                }
            }
        )

        # Get validated JSON from the validator
        validated_responses = result.get("json_validator", {}).get("validated", [])

        if validated_responses:
            # Get the first (and should be only) validated response
            chosen_data = json.loads(validated_responses[0].text)
            chosen_question_number = chosen_data.get("chosen_question_number")
            contextualized_question = chosen_data.get("contextualized_question")

            logger.info(
                f"LLM chose question with question_number: {chosen_question_number}"
            )
            logger.info(f"LLM contextualized question: {contextualized_question}")

            return {
                "chosen_question_number": chosen_question_number,
                "contextualized_question": contextualized_question,
            }
        else:
            # No valid JSON response was produced
            logger.warning("LLM failed to produce valid JSON response. Using fallback.")

    except Exception as e:
        logger.error(f"Unexpected error in pipeline execution: {e}")

    # Fallback logic (same as before)
    if remaining_questions:
        chosen_question_number = remaining_questions[0]["question_number"]
        contextualized_question = remaining_questions[0]["content"]
        logger.warning(
            f"Falling back to first remaining question: question_number {chosen_question_number}"
        )
        return {
            "chosen_question_number": chosen_question_number,
            "contextualized_question": contextualized_question,
        }
    else:
        logger.error("No remaining questions available to fall back on.")
        return None


def run_onboarding_data_extraction_pipeline(
    pipeline: Pipeline,
    user_response: str,
    user_context: dict[str, any],
    chat_history: list[str],
) -> dict[str, any]:
    """
    Run the onboarding data extraction pipeline and return extracted data.

    Args:
        pipeline: The configured pipeline
        user_response: User's latest message
        user_context: Previously collected user data
        chat_history: List of chat history messages

    Returns:
        Dictionary containing extracted data points
    """
    try:
        result = pipeline.run(
            {
                "user_response": user_response,
                "user_context": user_context,
                "chat_history": chat_history,
            }
        )

        llm_response = result.get("llm", {})
        replies = llm_response.get("replies", [])
        if replies:
            first_reply = replies[0]
            tool_calls = getattr(first_reply, "tool_calls", []) or []

            if tool_calls:
                arguments = getattr(tool_calls[0], "arguments", {})
                if isinstance(arguments, dict):
                    validated_args = {}
                    tool_schema = create_onboarding_tool().parameters
                    schema_props = tool_schema.get("properties", {})

                    for key, value in arguments.items():
                        if key in schema_props:
                            prop_details = schema_props[key]
                            if "enum" in prop_details:
                                if value in prop_details["enum"]:
                                    validated_args[key] = value
                            else:
                                validated_args[key] = value
                        else:
                            validated_args[key] = value
                    logger.info(f"Validated extracted data: {validated_args}")
                    return validated_args
                else:
                    logger.warning("Tool arguments are not a dictionary.")
                    return {}
            else:
                logger.warning("No tool calls found in LLM response")
                return {}
        else:
            logger.warning("No replies found in LLM response")
            return {}

    except Exception as e:
        logger.error(f"Error running extraction pipeline: {e}")
        return {}


def run_assessment_contextualization_pipeline(
    pipeline: Pipeline, flow_id: str, question_number: int, user_context: dict[str, any]
) -> str | None:
    """
    Run the assessment contextualization pipeline to get a contextualized question.

    Args:
        pipeline: The configured pipeline
        flow_id: The ID of the assessment flow
        question_number: The question number to contextualize
        user_context: Previously collected user data

    Returns:
        Contextualized question string or None if an error occurs
    """
    try:
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
        result = pipeline.run(
            {
                "retriever": {"filters": filters},
                "prompt_builder": {"user_context": user_context},
            }
        )

        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies"):
            return llm_response["replies"][0].text
        else:
            logger.warning("No replies found in LLM response for contextualization")
            return None

    except Exception as e:
        logger.error(f"Error running assessment contextualization pipeline: {e}")
        return None


def run_assessment_response_validator_pipeline(
    pipeline: Pipeline, user_response: str
) -> str | None:
    """
    Run the assessment response validator pipeline to validate a user's response.

    Args:
        pipeline: The configured pipeline
        user_response: User's response to validate

    Returns:
        Validated response string or "nonsense" if invalid
    """
    try:
        result = pipeline.run({"prompt_builder": {"user_response": user_response}})

        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies"):
            return llm_response["replies"][0].text
        else:
            logger.warning("No replies found in LLM response for validation")
            return None

    except Exception as e:
        logger.error(f"Error running assessment response validation pipeline: {e}")
        return None
