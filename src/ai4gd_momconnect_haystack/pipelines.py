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

SURVEY_NAVIGATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "next_step": {
            "type": "string",
            "description": "A short identifier for the next logical step in the survey (e.g., 'start', 'Q_seen', 'Q_bp', 'intent', 'thanks'). This MUST match a title in the source content.",
        },
        "is_final_step": {
            "type": "boolean",
            "description": "True if the survey is complete according to the logic, otherwise false.",
        },
    },
    "required": ["next_step", "is_final_step"],
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


def create_clinic_visit_tool() -> Tool:
    """Creates the data extraction tool for the ANC survey."""
    tool = Tool(
        name="extract_clinic_visit_data",
        description="Extracts structured data from a user's message about their antenatal clinic visit. Use this to understand if they went, who they saw, their experience, and any other relevant details.",
        function=extract_clinic_visit_data,
        parameters={
            "type": "object",
            "properties": {
                "visit_status": {
                    "type": "string",
                    "description": "Whether the user attended their clinic appointment.",
                    "enum": ["Yes, I went", "No, I'm not going", "I'm going soon"],
                },
                "reason_for_not_attending": {
                    "type": "string",
                    "description": "If the user did not attend and is not going, the reason why.",
                    "enum": [
                        "Didn't know about it 📅",
                        "Didn't know where 📍",
                        "Don't want check-ups ⛔",
                        "Can't go when open 🏥",
                        "Asked to pay 💰",
                        "Wait times too long ⌛",
                        "No support 🤝",
                        "Getting there is hard 🚌",
                        "I forgot 😧",
                        "Something else 😞",
                    ],
                },
                "reason_for_not_attending_other": {
                    "type": "string",
                    "description": "If the reason for not attending is 'Something else 😞', the user can specify the reason here.",
                },
                "professional_seen": {
                    "type": "string",
                    "description": "Whether the user saw a health professional, if they visited the clinic.",
                    "enum": ["Yes", "No"],
                },
                "reason_for_not_seeing_professional": {
                    "type": "string",
                    "description": "The reason why the user did not see a health professional, if they visited the clinic.",
                    "enum": [
                        "Clinic was closed ⛔",
                        "Wait time too long ⌛",
                        "No maternity record 📝",
                        "Asked to pay 💰",
                        "Told to come back 📅",
                        "Staff disrespectful 🤬",
                        "Something else 😞",
                    ],
                },
                "reason_for_not_seeing_professional_other": {
                    "type": "string",
                    "description": "If the reason for not seeing a health professional is 'Something else 😞', the user can specify the reason here.",
                },
                "blood_pressure_taken": {
                    "type": "string",
                    "description": "Whether the user's blood pressure was taken during the visit, if the user was seen by a health professional",
                    "enum": ["Yes", "No", "I don't know"],
                },
                "overall_experience_at_check_up": {
                    "type": "string",
                    "description": "The user's overall experience at the clinic, if the user was seen by a health professional",
                    "enum": ["Very good", "Good", "OK", "Bad", "Very bad"],
                },
                "good_experience_challenge": {
                    "type": "string",
                    "description": "The challenges the user faced during their clinic visit, if they had an OK or better experience.",
                    "enum": [
                        "No problems 👌",
                        "No maternity record 📝",
                        "Shamed/embarrassed 😳",
                        "No privacy 🤐",
                        "Not enough info ℹ️",
                        "Staff disespectful 🤬",
                        "Asked to pay 💰",
                        "Waited a long time ⌛",
                        "Something else 😞",
                    ],
                },
                "good_experience_challenge_other": {
                    "type": "string",
                    "description": "If the user had an OK or better experience and shared that they faced the challenge 'Something else 😞', the user can specify the challenge here.",
                },
                "bad_experience_challenge": {
                    "type": "string",
                    "description": "The challenges the user faced during their clinic visit, if they had a bad or very bad experience.",
                    "enum": [
                        "No maternity record 📝",
                        "Shamed/embarrassed 😳",
                        "No privacy 🤐",
                        "Not enough info ℹ️",
                        "Staff disrespectful 🤬",
                        "Asked to pay 💰",
                        "Waited a long time ⌛",
                        "Something else 😞",
                    ],
                },
                "bad_experience_challenge_other": {
                    "type": "string",
                    "description": "If the user had a bad or very bad experience and shared that they faced the challenge 'Something else 😞', the user can specify the challenge here.",
                },
                "biggest_challenge_of_the_visit": {
                    "type": "string",
                    "description": "The biggest challenge the user faced in getting to their clinic visit, given that they attended and were seen by a health professional.",
                    "enum": [
                        "No challenges 👌",
                        "Transport 🚌",
                        "No support 🤝",
                        "Clinic opening hours 🏥",
                        "Something else 😞",
                    ],
                },
                "biggest_challenge_of_the_visit_other": {
                    "type": "string",
                    "description": "If the user's biggest challenge in getting to their clinic visit is 'Something else 😞', the user can specify the challenge here.",
                },
                "intention_to_attend_next_visit": {
                    "type": "string",
                    "description": "Whether the user intends to attend their next clinic visit, regardless of whether they attended the latest one.",
                    "enum": ["Yes, I will", "No, I won't"],
                },
                "survey_difficulty": {
                    "type": "string",
                    "description": "If the user found this survey easy or difficult to answer, given that they haven't completed it before.",
                    "enum": [
                        "Very easy",
                        "A little easy",
                        "OK",
                        "A little difficult",
                        "Very difficult",
                    ],
                },
            },
            "additionalProperties": {
                "type": "string",
                "description": "Any other valuable information related to the clinic visit, like tests done, advice received, or specific concerns.",
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
    Data already collected:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Remaining questions to complete their profile:
    {% for q in remaining_questions %}
    - Question: "{{ q.content }}" (collects data for: "{{ q.collects }}", possible correctly formatted values: "{{ q.valid_responses }}", current question_number for reference: {{ q.question_number }})
    {% endfor %}

    Considering the information already collected, the chat history, and the remaining questions,
    which single question would be the most natural and effective to ask next?

    You MUST respond with a valid JSON object containing exactly these fields:
    - "chosen_question_number" (integer): The question_number of the chosen question from the list above
    - "contextualized_question" (string): A contextualized version of the question

    You can reference the existing User Context and chat history to modify the tonality and/or phrasing,
    but DO NOT change the core meaning of the question nor introduce ambiguity. Ensure that the chat flows
    smoothly (e.g. the first message in a chat must not start as if there were preceding messages).

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
    Data already collected:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    User's latest message: "{{ user_response }}"

    Please use the 'extract_onboarding_data' tool to analyze this message. Pay close attention to these **critical rules**:
    - For the properties 'province', 'area_type', 'relationship_status', 'education_level', 'hunger_days', 'num_children' and 'phone_ownership', extracted data **MUST** adhere strictly to their 'enum' lists. If the user's response for one of these properties does **NOT** contain a word or phrase that *directly and unambiguously* maps to one of the EXACT 'enum' values, **DO NOT include that property in your tool call**. Only store the 'Skip' enum value for these properties if the user explicitly states they want to skip in response to that specific question.
    - **DO NOT GUESS or INFER** an enum value based on sentiment, vague descriptions, or ambiguous terms. Only include a field if you are highly confident that the user's input matches an allowed 'enum' value.
    - Do not extract a data point if it clearly has already been collected in the user context, unless the user's latest message explicitly provides new information that updates it.
    - For properties with numeric ranges like 'hunger_days', you MUST map the user's input to the correct enum category, unless they did not provide valid information. Do not just look for an exact string match. For example:
        - If the user says "3", you should extract: {"hunger_days": "3-4 days"}
        - If the user says "one day", you should extract: {"hunger_days": "1-2 days"}
        - If the user says "6", you should extract: {"hunger_days": "5-7 days"}
        - If the user says "I haven't been hungry", you should extract: {"hunger_days": "0 days"}
        - If the user does not provide a mappable, do not include 'hunger_days' in the tool call.
    - For 'num_children', apply the same numeric mapping logic.
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
        template=[ChatMessage.from_user(prompt_template)],
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
    against a dynamic list of valid responses.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator is not available. Cannot create Assessment Response Validation Pipeline."
        )
        return None
    pipeline = Pipeline()

    prompt_template = """
    You are an assistant validating a user's response to a question. The only valid possible responses are in the list below.
    Your task is to determine if the user's answer maps to one of these valid options.

    Valid Responses:
    {% for response in valid_responses %}
    - "{{ response }}"
    {% endfor %}

    - If the user's response clearly and unambiguously corresponds to one of the valid responses, your output MUST be the exact text of that valid response from the list.
    - If the user's response is ambiguous, does not match any valid response, or is nonsense/gibberish, you MUST return the single word: "nonsense".

    User Response:
    {{ user_response }}

    Validated Response:
    """
    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_response", "valid_responses"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created Assessment Response Validation Pipeline.")
    return pipeline


@cache
def create_clinic_visit_navigator_pipeline() -> Pipeline | None:
    """
    Creates a pipeline that determines the next logical step in the dynamic ANC survey.
    """
    llm_generator = get_llm_generator()
    if not llm_generator:
        logger.error(
            "LLM Generator not available. Cannot create Clinic Visit Navigator Pipeline."
        )
        return None

    pipeline = Pipeline()

    prompt_template = """
    You are a survey logic engine. Your task is to determine the next step in a dynamic antenatal clinic visit survey based on the user's data.

    **Survey Logic:**
    Your decision-making process must follow this logic precisely:
    1.  **Start:** If no data is present, the first step is always 'start'.
    2.  **Branch on `visit_status`:**
        * "No, I'm not going": Next is 'Q_why_not_go'. Then 'intent'. Then 'thanks'.
        * "I'm going soon": Next is 'start_going_soon'. Then survey is complete.
        * "Yes, I went": Next is 'Q_seen'.
    3.  **Branch on `professional_seen`:**
        * "No": Next is 'Q_why_no_visit'. Then 'intent'. Then 'thanks'.
        * "Yes": Next is 'Q_bp'. Then 'Q_experience'.
    4.  **Branch on `overall_experience_at_check_up`:**
        * "Very good", "Good", "OK": Next is 'Q_visit_good'.
        * "Bad", "Very bad": Next is 'bad', then 'Q_visit_bad'.
    5.  **After Experience Branch:** The step after either `Q_visit_good` or `Q_visit_bad` is `Q_challenges`.
    6.  **Final Steps:** After `Q_challenges`, the next step is `intent`. After `intent`, the next step is `thanks`. After `thanks`, the survey is complete.

    **Data Collected So Far (User Context):**
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Based on the survey logic and the data collected so far, determine the `next_step` identifier.
    - If all necessary questions for a user's path have been asked, set `is_final_step` to true. The final step before completion is always 'thanks'.
    - The `next_step` value MUST be a valid title from the survey content (e.g., 'start', 'Q_seen', 'Q_bp').

    You MUST respond with a valid JSON object following this schema:
    - "next_step" (string): The identifier for the next logical step.
    - "is_final_step" (boolean): Set to true ONLY if the survey is now complete.

    JSON Response:
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context"],
    )

    json_validator = JsonSchemaValidator(json_schema=SURVEY_NAVIGATOR_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created Clinic Visit Navigator Pipeline.")
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
    You are a conversational assistant for a maternal health chatbot.
    Your task is to take a standard survey question and make it sound natural and conversational for the user,
    WITHOUT changing the core meaning of the question or introducing ambiguity.

    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Chat History:
    {% for message in chat_history %}
    - {{ message }}
    {% endfor %}

    Original Question to send:
    "{{ original_question }}"

    Valid Responses:
    {% for vr in valid_responses %}
    {{ vr }}
    {% endfor %}

    Please rephrase the "Original Question" to be more personal and friendly.
    - You can use information from the user context (like their name, if available).
    - Add emojis where appropriate to maintain a warm tone.
    - If the original question is already good enough, you can return it as is.
    - If a list of valid responses is supplied above, ensure that the new contextualized
      question is phrased such that the valid responses still make sense, and would be
      grammatically correct.

    You MUST respond with a valid JSON object with one key: "contextualized_question".

    JSON Response:
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=[
            "user_context",
            "chat_history",
            "original_question",
            "valid_responses",
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


@cache
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

    extraction_tool = create_clinic_visit_tool()
    llm_generator.tools = [extraction_tool]

    pipeline = Pipeline()

    prompt_template = """
    You are an assistant helping to extract structured information from a user's response to a question about their antenatal clinic (ANC) visit.

    Data already collected during this survey:
    {{ user_context }}

    Chat history:
    {{ chat_history }}

    User's latest message: "{{ user_response }}"

    Please use the 'extract_clinic_visit_data' tool to analyze the user's latest message. Your primary goal is to extract the specific pieces of information the user has provided in this message.
    - Only extract values that the user has clearly stated.
    - Match the user's response to the correct 'enum' values where applicable.
    - If the user provides a reason that matches 'Something else 😞', capture their specific reason in the corresponding '_other' field (e.g., 'reason_for_not_attending_other').
    - Do not guess or infer information not present in the user's latest message.
    """

    prompt_builder = ChatPromptBuilder(
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["user_context", "chat_history", "user_response"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info("Created ANC Survey Data Extraction Pipeline with Tools.")
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

    Please classify the user's message into one of these intents:
    - 'JOURNEY_RESPONSE': The user is directly answering or attempting to answer the question asked.
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
        template=[ChatMessage.from_user(prompt_template)],
        required_variables=["last_question", "user_response"],
    )

    json_validator = JsonSchemaValidator(json_schema=INTENT_DETECTION_SCHEMA)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    pipeline.add_component("json_validator", json_validator)

    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "json_validator.messages")

    logger.info("Created Intent Detection Pipeline.")
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
        template=[ChatMessage.from_user(prompt_template)],
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

        # Get the original document to access the valid_responses
        retrieved_docs = result.get("retriever", {}).get("documents", [])
        if not retrieved_docs:
            logger.error(
                f"Could not retrieve document for flow '{flow_id}' and question {question_number}."
            )
            return None

        valid_responses = retrieved_docs[0].meta.get("valid_responses", [])
        print(f"Valid responses for question {question_number}: {valid_responses}")

        # Get validated JSON from the validator
        validated_json_list = result.get("json_validator", {}).get("validated", [])
        if validated_json_list:
            # Get the contextualized question from the first valid JSON object
            question_data = json.loads(validated_json_list[0].text)
            contextualized_question_text = question_data.get("contextualized_question")

            final_question = f"{contextualized_question_text}\n\n" + "\n".join(
                valid_responses
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
        result = pipeline.run(
            {
                "prompt_builder": {
                    "user_response": user_response,
                    "valid_responses": valid_responses,
                }
            }
        )

        llm_response = result.get("llm", {})
        if llm_response and llm_response.get("replies"):
            validated_text = llm_response["replies"][0].text.strip().strip('"')
            if validated_text.lower() == "nonsense":
                return None
            if validated_text in valid_responses:
                return validated_text
            else:
                logger.warning(
                    f"LLM returned a response ('{validated_text}') not in the valid list. Treating as invalid."
                )
                return None
        else:
            logger.warning("No replies found in LLM response for validation")
            return None

    except Exception as e:
        logger.error(f"Error running assessment response validation pipeline: {e}")
        return None


def run_clinic_visit_navigator_pipeline(
    pipeline: Pipeline, user_context: dict[str, Any]
) -> dict | None:
    """
    Runs the clinic visit navigator pipeline to determine the next survey step.
    Returns the step identifier, not the full question.
    """
    try:
        result = pipeline.run(
            {
                "prompt_builder": {
                    "user_context": user_context,
                }
            }
        )
        validated_responses = result.get("json_validator", {}).get("validated", [])
        if validated_responses:
            next_step_data = json.loads(validated_responses[0].text)
            logger.info(f"LLM chose next survey step: {next_step_data}")
            return next_step_data
        else:
            logger.warning("Navigator pipeline failed to produce valid JSON response.")
            return None
    except Exception as e:
        logger.error(f"Error running clinic visit navigator pipeline: {e}")
        return None


def run_anc_survey_contextualization_pipeline(
    pipeline: Pipeline,
    user_context: dict[str, Any],
    chat_history: list[str],
    original_question: str,
    valid_responses: list[str],
) -> str | None:
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
    user_context: dict[str, Any],
    chat_history: list[str],
) -> dict[str, Any]:
    """
    Run the ANC survey data extraction pipeline and return extracted data.

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
                    logger.info(f"Extracted ANC data: {arguments}")
                    return arguments
            else:
                logger.warning("No tool calls found in LLM response for ANC survey")
                return {}
        else:
            logger.warning("No replies found in LLM response for ANC survey")
            return {}

    except Exception as e:
        logger.error(f"Error running ANC survey extraction pipeline: {e}")
        return {}
    return {}


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
