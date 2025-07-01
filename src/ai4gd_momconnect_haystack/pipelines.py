import json
import logging
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

# --- The ANC Survey flow, statically defined ---
ANC_SURVEY_FLOW_LOGIC = {
    "start": lambda ctx: "Q_seen"
    if ctx.get("start") == "Yes, I went"
    else "start_not_going"
    if ctx.get("start") == "No, I'm not going"
    else "start_going_soon"
    if ctx.get("start") == "I'm going soon"
    else None,
    # I'm going soon
    "start_going_soon": lambda ctx: "__GOING_SOON_REMINDER_3_DAYS__",
    # Yes, I went
    "Q_seen": lambda ctx: "seen_yes"
    if ctx.get("Q_seen") == "Yes"
    else "Q_seen_no"
    if ctx.get("Q_seen") == "No"
    else None,
    "seen_yes": lambda ctx: "Q_bp"
    if ctx.get("seen_yes") == "Yes"
    else "mom_ANC_remind_me_01"
    if ctx.get("seen_yes") == "Remind me tomorrow"
    else None,
    "Q_seen_no": lambda ctx: "Q_why_no_visit"
    if ctx.get("Q_seen_no") == "Yes"
    else "mom_ANC_remind_me_01"
    if ctx.get("Q_seen_no") == "Remind me tomorrow"
    else None,
    "Q_why_no_visit": lambda ctx: "intent"
    if ctx.get("Q_why_no_visit")
    in [
        "Clinic was closed ⛔",
        "Wait time too long ⌛",
        "No maternity record 📝",
        "Asked to pay 💰",
        "Told to come back 📅",
        "Staff disrespectful 🤬",
    ]
    else "Q_why_no_visit_other"
    if ctx.get("Q_why_no_visit") == "Something else 😞"
    else None,
    "Q_why_no_visit_other": lambda ctx: "intent",
    "mom_ANC_remind_me_01": lambda ctx: "__NOT_GOING_REMINDER_1_DAY__",
    "Q_bp": lambda ctx: "Q_experience",
    "Q_experience": lambda ctx: "bad"
    if ctx.get("Q_experience") in ["Bad", "Very bad"]
    else "good"
    if ctx.get("Q_experience") in ["Very good", "Good", "OK"]
    else None,
    "bad": lambda ctx: "Q_visit_bad",
    "good": lambda ctx: "Q_visit_bad",
    "Q_visit_good": lambda ctx: "Q_challenges"
    if ctx.get("Q_visit_good")
    in [
        "No problems 👌",
        "No maternity record 📝",
        "Shamed/embarrassed 😳",
        "No privacy 🤐",
        "Not enough info ℹ️",
        "Staff disespectful 🤬",
        "Asked to pay 💰",
        "Waited a long time ⌛",
    ]
    else "Q_visit_other"
    if ctx.get("Q_visit_good") == "Something else 😞"
    else None,
    "Q_visit_bad": lambda ctx: "Q_challenges"
    if ctx.get("Q_visit_bad")
    in [
        "No maternity record 📝",
        "Shamed/embarrassed 😳",
        "No privacy 🤐",
        "Not enough info ℹ️",
        "Staff disrespectful 🤬",
        "Asked to pay 💰",
        "Waited a long time ⌛",
    ]
    else "Q_visit_other"
    if ctx.get("Q_visit_bad") == "Something else 😞"
    else None,
    "Q_visit_other": lambda ctx: "Q_challenges",
    "Q_challenges": lambda ctx: "intent"
    if ctx.get("Q_challenges")
    in ["No challenges 👌", "Transport 🚌", "No support 🤝", "Clinic opening hours 🏥"]
    else "Q_challenges_other"
    if ctx.get("Q_challenges") == "Something else 😞"
    else None,
    "Q_challenges_other": lambda ctx: "intent",
    # No, I'm not going
    "start_not_going": lambda ctx: "Q_why_not_go"
    if ctx.get("start_not_going") == "Yes"
    else "mom_ANC_remind_me_02"
    if ctx.get("start_not_going") == "Remind me tomorrow"
    else None,
    "mom_ANC_remind_me_02": lambda ctx: "__WENT_REMINDER_1_DAY__",
    "Q_why_not_go": lambda ctx: "intent"
    if ctx.get("Q_why_not_go")
    in [
        "Didn't know about it 📅",
        "Didn't know where 📍",
        "Don't want check-ups ⛔",
        "Can't go when open 🏥",
        "Asked to pay 💰",
        "Wait times too long ⌛",
        "No support 🤝",
        "Getting there is hard 🚌",
        "I forgot 😧",
    ]
    else "Q_why_not_go_other"
    if ctx.get("Q_why_not_go") == "Something else 😞"
    else None,
    "Q_why_not_go_other": lambda ctx: "intent",
    "intent": lambda ctx: "end"
    if (not ctx.get("first_survey")) and ctx.get("intent") == "Yes, I will"
    else "feedback_if_first_survey"
    if ctx.get("first_survey") and ctx.get("intent") == "Yes, I will"
    else "not_going_next_one"
    if ctx.get("intent") == "No, I won't"
    else None,
    "not_going_next_one": lambda ctx: "end"
    if not ctx.get("first_survey")
    else "feedback_if_first_survey"
    if ctx.get("first_survey")
    else None,
    # Feedback and thanks after the user's first survey completion
    "feedback_if_first_survey": lambda ctx: "end_if_feedback",
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
{% for message in chat_history %}
{% if message.is_from('user') %}
user:
{{ message.text }}

{% elif message.is_from('assistant') %}
assistant:
{{ message.text }}

{% else %}
system:
{{ message.text }}

{% endif %}
{% endfor %}

system:
User Context:
{% for key, value in user_context.items() %}
- "{{ key }}": "{{ value }}"
{% endfor %}

Remaining questions to complete user profile:
{% for q in remaining_questions %}
Question {{ q.question_number }}: "{{ q.content }}" (with valid possible responses: "{{ q.valid_responses }}")
{% endfor %}

Your task is to select the single remaining question that would be the most natural and effective to ask next and contextualize it if you think it's needed.
- You can reference the existing chat history and user context above to modify the tonality and/or phrasing for the user, but DO NOT change the core meaning of the question or introduce ambiguity.
- DO NOT list the valid responses in the contextualized question.
- Ensure that the dialogue flows smoothly (e.g. the first message in a chat must not start as if there were preceding messages), and to still ask the question in such a way that its valid responses will remain grammatically valid responses to the new version of the question.

You MUST respond with a valid JSON object containing exactly these fields:
- "chosen_question_number" (integer): The question_number of the chosen question from the list above.
- "contextualized_question" (string): Your version of the question you chose.

JSON Response:
    """
    chat_template = [
        ChatMessage.from_system(prompt_template),
    ]
    prompt_builder = ChatPromptBuilder(
        template=chat_template,
        required_variables=["user_context", "remaining_questions", "chat_history"],
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
{% for message in chat_history %}
{% if message.is_from('user') %}
user:
{{ message.text }}

{% elif message.is_from('assistant') %}
assistant:
{{ message.text }}

{% else %}
system:
{{ message.text }}

{% endif %}
{% endfor %}

system:
User Context:
{% for key, value in user_context.items() %}
- {{ key }}: {{ value }}
{% endfor %}

User's latest message:
"{{ user_response }}"

Please use the 'extract_onboarding_data' tool to analyze the user's latest message and extract their intended response:
- For the properties 'province', 'area_type', 'relationship_status', 'education_level', 'hunger_days', 'num_children' and 'phone_ownership', extracted data **MUST** adhere strictly to their 'enum' lists. If the user's response for one of these properties does **NOT** contain a word or phrase that *directly and unambiguously* maps to one of the EXACT 'enum' values, **DO NOT include that property in your tool call**. Only store the 'Skip' enum value for these properties if the user explicitly states they want to skip.
- **DO NOT GUESS or INFER** an enum value based on sentiment, vague descriptions, or ambiguous terms. Only include a field if you are highly confident that the user's input matches an allowed 'enum' value.
- Do not extract a data point if it clearly has already been collected in the user context, unless the user's latest message explicitly provides new information that updates it.
- For properties with numeric ranges like 'hunger_days', you MUST map the user's input to the correct enum category whose range contains or corresponds to the user's input, unless they did not provide valid information. Do not just look for an exact string match. As examples:
    - If the user says "3", you should extract: {"hunger_days": "3-4 days"}
    - If the user says "one day", you should extract: {"hunger_days": "1-2 days"}
    - If the user says "6", you should extract: {"hunger_days": "5-7 days"}
    - If the user says "I haven't been hungry", you should extract: {"hunger_days": "0 days"}
    - If the user does not provide a mappable, do not include 'hunger_days' in the tool call.
- For 'num_children', apply similar numeric mapping logic.
- For the open-ended additionalProperties, extract any extra information mentioned that is not already in one of the expected properties, and AS LONG AS it pertains specifically to maternal health or the use of a maternal health chatbot.
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
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

Original Assessment Question to be contextualized:
{{ documents[0].content }}

Valid responses:
{% for valid_response in documents[0].meta.valid_responses %}
- "{{ valid_response }}"
{% endfor %}

Review the Original Assessment Question. If you think it's needed, make minor
adjustments to ensure that the question is clear and directly applicable to the
user's context. **Crucially, do not change the core meaning, difficulty, or the
scale/format of the question.** If no changes are needed, return the original question text.

You MUST respond with a valid JSON object containing exactly one key:
"contextualized_question". The value should be the question text ONLY. DO NOT
include the list of possible answers in your response, but make sure that the
contextualized question is asked in such a way that the user can still
respond with one of the valid responses listed above.

JSON Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=["user_context", "documents"],
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

    prompt_template = """
You are an AI assistant validating a user's response to an assessment question in a chatbot for new mothers in South Africa.
Your task is to analyze the user's response and determine if it maps to one of the allowed responses provided below.

Allowed Responses:
{% for response in valid_responses %}
- "{{ response }}"
{% endfor %}

User Response:
"{{ user_response }}"

You MUST respond with a valid JSON object. The JSON should contain a single key, "validated_response".

- If the user's response clearly and unambiguously corresponds to one of the "Allowed Responses", the value of "validated_response" should be the exact text of that allowed response.
- If the user's response is ambiguous, does not match any of the allowed responses, or is nonsense/gibberish, you MUST set the value of "validated_response" to "nonsense".

JSON Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=["user_response", "valid_responses", "previous_message"],
    )

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info(
        "Created Assessment Response Validation Pipeline with JSON Schema validation."
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

    prompt_template = """
You are an AI assistant validating a user's response to the previous message sent by a chatbot for new mothers in South Africa.
Your task is to analyze the user's response and determine if it maps to one of the allowed responses provided below.

Allowed Responses:
{% for response in valid_responses %}
- "{{ response }}"
{% endfor %}

Previous Message:
"{{ previous_message }}"

User Response:
"{{ user_response }}"

You MUST respond with a valid JSON object. The JSON should contain a single key, "validated_response".

- If the user's response clearly and unambiguously corresponds to one of the "Allowed Responses", the value of "validated_response" should be the exact text of that allowed response.
- If the user's response is ambiguous, does not match any of the allowed responses, or is nonsense/gibberish, you MUST set the value of "validated_response" to "nonsense".

JSON Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=["user_response", "valid_responses", "previous_message"],
    )

    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", prompt_builder)
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

    prompt_template = """
{% for message in chat_history %}
{% if message.is_from('user') %}
user:
{{ message.text }}

{% elif message.is_from('assistant') %}
assistant:
{{ message.text }}

{% else %}
system:
{{ message.text }}

{% endif %}
{% endfor %}

system:
Your task is to take the next survey question and contextualize it for the user, WITHOUT changing the core meaning of the question or introducing ambiguity.

User Context:
{% for key, value in user_context.items() %}
- {{ key }}: {{ value }}
{% endfor %}

Next survey question:
"{{ original_question }}"

{% if valid_responses %}
Allowed Responses:
{% for vr in valid_responses %}
- "{{ vr }}"
{% endfor %}
{% endif %}

You MUST respond with a valid JSON object with one key: "contextualized_question".

Rephrase the survey question if you think it's needed and return it as the "contextualized_question".
- You can use information from the user context
- If the survey question is already good enough, you can return it as is
{% if valid_responses %}
- If a list of allowed responses is supplied above, ensure that the new contextualized question is phrased such that the allowed responses would still make sense, and would be grammatically correct in dialogue.
{% endif %}

JSON Response:
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=[
            "user_context",
            "chat_history",
            "original_question",
        ],
    )

    json_validator = JsonSchemaValidator(json_schema=SURVEY_QUESTION_CONTEXT_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
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

    prompt_template = """
You are an AI assistant helping to extract information from a user's (a new South African mother) response to a pregnancy clinic visit survey question/message into a structured format.

Previous survey question/message:
"{{ previous_service_message }}"

User's latest response:
- "{{ user_response }}"

You MUST respond with a valid JSON object. The JSON should contain a single key, "validated_response".

Your task is to analyze the user's response *in light of the previous survey question/message and its expected responses* and determine which of the expected responses it maps to in meaning and intent.
- If the user response to the previous survey question/message maps to one of the expected responses stated in the previous survey question/message, the value you return in "validated_response" must be the exact text of that matched expected response. The user's response need not be an exact string match of an expected response since they might be using slang, colloquiolisms, synonyms or shortened versions of the response, as long as it maps in meaning and intent, e.g.:
  - Users might respond with "OK" or "Okay" when they mean "Yes" or "Continue"
  - Users might respond with shortened versions of the expected responses, e.g. "Yes" instead of "Yes, I will"
  - Users might respond with "something else" or "other" when they mean "Something else 😞"
- If the user response does not map to an expected response in intent or meaning, or it is nonsense/gibberish, you must set the value of "validated_response" to "nonsense".

JSON Response:
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=["previous_service_message", "user_response"],
    )
    json_validator = JsonSchemaValidator()

    pipeline.add_component("prompt_builder", prompt_builder)
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

    prompt_template = """
You are an intent classifier for a maternal health chatbot. Your task is to analyze the user's latest message and classify it into ONE of the following categories based on its intent.

Last question the user was asked:
"{{ last_question }}"

User's response:
"{{ user_response }}"

Please classify the user's message, in light of the last question sent to the user, into one of these intents:
- 'JOURNEY_RESPONSE': The user is directly answering, attempting to answer, or skipping the question asked.
- 'QUESTION_ABOUT_STUDY': The user is asking a question about the research study itself (e.g., "who are you?", "why are you asking this?").
- 'HEALTH_QUESTION': The user is asking a new question related to health, pregnancy, or their wellbeing, instead of answering the question.
- 'ASKING_TO_STOP_MESSAGES': The user expresses a desire to stop receiving messages.
- 'ASKING_TO_DELETE_DATA': The user wants to leave the study and have their data deleted.
- 'REPORTING_AIRTIME_NOT_RECEIVED': The user is reporting that they have not received their airtime incentive.
- 'CHITCHAT': The user is making a conversational comment that is not a direct answer, a question, or a request.

You MUST respond with a valid JSON object containing exactly one key: "intent".

JSON Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
        required_variables=["last_question", "user_response"],
    )

    json_validator = JsonSchemaValidator(json_schema=INTENT_DETECTION_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

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

    prompt_template = """
You are a helpful assistant for a maternal health chatbot. Answer the user's question based ONLY on the provided context information.
If the context does not contain the answer, say that you do not have that information.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

User's Question: {{ user_question }}

Answer:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_system(prompt_template)],
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


# --- Running Pipelines ---
def run_next_onboarding_question_pipeline(
    pipeline: Pipeline,
    user_context: dict[str, Any],
    remaining_questions: list[dict],
    chat_history: list[ChatMessage],
) -> dict[str, Any] | None:
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

        validated_responses = result.get("json_validator", {}).get("validated", [])

        if validated_responses:
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
            logger.warning("LLM failed to produce valid JSON response. Using fallback.")

    except Exception as e:
        logger.error(f"Unexpected error in pipeline execution: {e}")

    # Fallback logic
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
    user_context: dict[str, Any],
    chat_history: list[str],
) -> dict[str, Any]:
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
                "prompt_builder": {
                    "user_response": user_response,
                    "user_context": user_context,
                    "chat_history": chat_history,
                }
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
    pipeline: Pipeline, flow_id: str, question_number: int, user_context: dict[str, Any]
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
            },
            include_outputs_from=["retriever"],
        )

        retrieved_docs = result.get("retriever", {}).get("documents", [])
        if not retrieved_docs:
            logger.error(
                f"Could not retrieve document for flow '{flow_id}' and question {question_number}."
            )
            return None

        valid_responses = retrieved_docs[0].meta.get("valid_responses", [])

        validated_json_list = result.get("json_validator", {}).get("validated", [])
        if validated_json_list:
            question_data = json.loads(validated_json_list[0].text)
            contextualized_question_text = question_data.get("contextualized_question")

            final_question = f"{contextualized_question_text}\n\n" + "\n".join(
                ["Please reply with one of the following:"]
                + [f"- '{vr}'" for vr in valid_responses]
            )
            return final_question.strip()

        else:
            logger.error("LLM failed to produce valid JSON.")
            return None

    except Exception as e:
        logger.error(f"Error running assessment contextualization pipeline: {e}")
        return None


def run_assessment_response_validator_pipeline(
    pipeline: Pipeline, user_response: str, valid_responses: list[str]
) -> str | None:
    """
    Run the assessment response validator pipeline to validate a user's response.

    Args:
        pipeline: The configured pipeline.
        user_response: User's response to validate.
        valid_responses: A list of valid string responses for the current question.

    Returns:
        The validated response string, or None if the response is invalid/nonsense.
    """
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
                    "user_response": user_response,
                    "valid_responses": valid_responses,
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        if (
            result
            and "json_validator" in result
            and result["json_validator"].get("validated")
        ):
            validated_message: ChatMessage = result["json_validator"]["validated"][0]
            validated_json = json.loads(validated_message.text)
            validated_response = validated_json.get("validated_response")

            if validated_response != "nonsense":
                return validated_response
            else:
                logger.warning(
                    f"User response '{user_response}' was validated as 'nonsense'."
                )
                return None
        else:
            logger.warning(
                f"Assessment response validation failed. LLM output did not match schema for user response: '{user_response}'"
            )
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running assessment end response validation pipeline: {e}")
        return None


def run_assessment_end_response_validator_pipeline(
    pipeline: Pipeline,
    user_response: str,
    valid_responses: list[str],
    previous_message: str,
) -> str | None:
    """
    Run the assessment end response validator pipeline to validate a user's response.

    Args:
        pipeline: The configured pipeline.
        user_response: User's response to validate.
        valid_responses: A list of valid string responses for the message.
        previous_message: The message to which the user is responding.

    Returns:
        The validated response string, or None if the response is invalid/nonsense.
    """
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
                    "user_response": user_response,
                    "valid_responses": valid_responses,
                    "previous_message": previous_message,
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        if (
            result
            and "json_validator" in result
            and result["json_validator"]["validated"]
        ):
            validated_json = result["json_validator"]["validated"][0].meta["parsed"]
            if validated_json["validated_response"] != "nonsense":
                return validated_json["validated_response"]
        else:
            logger.warning("Assessment end response validation failed.")
        return None

    except Exception as e:
        logger.error(f"Error running assessment end response validation pipeline: {e}")
        return None


def run_anc_survey_contextualization_pipeline(
    pipeline: Pipeline,
    user_context: dict[str, Any],
    chat_history: list[str],
    original_question: str,
    valid_responses: list[str],
) -> str:
    """
    Runs the ANC survey contextualization pipeline.
    """
    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "user_context": user_context,
                    "chat_history": chat_history,
                    "original_question": original_question,
                    "valid_responses": valid_responses,
                }
            }
        )
        validated_responses = result.get("json_validator", {}).get("validated", [])
        if validated_responses:
            question_data = json.loads(validated_responses[0].text)
            return question_data.get("contextualized_question")
        else:
            logger.warning(
                "Contextualization pipeline failed to produce valid JSON. Falling back to original question."
            )
            return original_question
    except Exception as e:
        logger.error(f"Error running ANC survey contextualization pipeline: {e}")
        return original_question  # Fallback


def run_clinic_visit_data_extraction_pipeline(
    pipeline: Pipeline,
    user_response: str,
    previous_service_message: str,
    valid_responses: list[str],
) -> str | None:
    """
    Run the ANC survey data extraction pipeline and return extracted data.

    Args:
        pipeline: The configured pipeline
        user_response: User's latest message
        previous_service_message: Previous message sent to the user, to which they are responding.
        valid_responses: A list of valid responses associated with the previous_service_message.

    Returns:
        String containing extracted data point
    """
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
                    "user_response": user_response,
                    "previous_service_message": previous_service_message,
                },
                "json_validator": {"json_schema": valid_responses_schema},
            }
        )

        if (
            result
            and "json_validator" in result
            and result["json_validator"].get("validated")
        ):
            validated_message: ChatMessage = result["json_validator"]["validated"][0]
            validated_json = json.loads(validated_message.text)
            validated_response = validated_json.get("validated_response")

            if validated_response != "nonsense":
                return validated_response
            else:
                logger.warning(
                    f"User response '{user_response}' was validated as 'nonsense'."
                )
                return None
        else:
            logger.warning(
                f"Clinic visit data extraction failed. LLM output did not match schema for user response: '{user_response}'"
            )
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running clinic visit data extraction pipeline: {e}")
        return None


def run_intent_detection_pipeline(
    pipeline: Pipeline, last_question: str, user_response: str
) -> dict[str, Any] | None:
    """
    Runs the intent detection pipeline.
    """
    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "last_question": last_question,
                    "user_response": user_response,
                }
            }
        )
        validated_responses = result.get("json_validator", {}).get("validated", [])
        if validated_responses:
            intent_data = json.loads(validated_responses[0].text)
            logger.info(f"LLM classified intent as: {intent_data}")
            return intent_data
        else:
            logger.warning("Intent pipeline failed to produce valid JSON response.")
            return None
    except Exception as e:
        logger.error(f"Error running intent detection pipeline: {e}")
        return None


def run_faq_pipeline(
    pipeline: Pipeline, user_question: str, filters: dict
) -> dict[str, Any] | None:
    """
    Runs the FAQ answering pipeline.
    """
    try:
        result = pipeline.run(
            {
                "retriever": {"filters": filters},
                "prompt_builder": {"user_question": user_question},
            }
        )
        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies"):
            answer = llm_response["replies"][0].text
            return {"answer": answer}
        else:
            logger.warning("No replies found in LLM response for FAQ pipeline")
            return None

    except Exception as e:
        logger.error(f"Error running FAQ pipeline: {e}")
        return None
