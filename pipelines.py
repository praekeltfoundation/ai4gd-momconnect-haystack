import logging

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import FilterRetriever
from haystack.utils import Secret

from doc_store import document_store


# --- Configuration ---
logger = logging.getLogger(__name__)

# --- Pipelines ---
def create_next_onboarding_question_pipeline() -> Pipeline | None:
    """
    Creates a pipeline where an LLM selects the best next onboarding question
    from a list of remaining questions, given the user's current context.
    """
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_generator = OpenAIGenerator(api_key=openai_api_key, model="gpt-3.5-turbo")
        logger.info("OpenAI Generator instance created for next question selection.")
    except ValueError:
        logger.error("OPENAI_API_KEY environment variable not found. Cannot create LLM component.")
        return None
    if not llm_generator:
        logger.error("LLM Generator not available. Cannot create next question selection pipeline.")
        return None

    pipeline = Pipeline()

    prompt_template = """
    You are an assistant helping to onboard a new user.
    The user has already provided the following information:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Here is their current chat history:
    Chat History:
    {% for message in chat_history %}
    - {{ message }}
    {% endfor %}

    Here are the remaining questions we can ask to complete their profile:
    Remaining Questions:
    {% for q in remaining_questions %}
    - Question: "{{ q.content }}" (collects data for: "{{ q.collects }}", possible correctly formatted values: "{{ q.valid_responses }}", current question_number for reference: {{ q.question_number }})
    {% endfor %}

    Considering the information already collected, the current chat history, and the remaining questions,
    which single question would be the most natural and effective to ask next?
    Your goal is to make the onboarding conversational and logical such that a subsequent prompt can be made to try to extract the expected data from the user's response.

    Respond with a JSON object containing the 'question_number' of the chosen question from the list above,
    along with a contextualized version of the question. You can reference the existing User Context and Chat History to
    modify the tonality and/or phrasing, or even adding emoji's, but do not change the core meaning of the question.
    For example, asking "Are you in a relationship?" could be contextualized to "Are you in a relationship? 💑" for
    a female if you think that it's appropriate. Ensure that the chat flows smoothly (e.g. the first message
    in a chat must not start as if there were preceding messages).
    Example Output:
    {
        "chosen_question_number": 3,
        "contextualized_question": "Are you in a relationship? 💑"
    }

    Chosen Question (JSON object with chosen_question_number and contextualized_question):
    """
    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["user_context", "remaining_questions", "chat_history"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator) # Use the existing llm_generator

    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    logger.info("Created Next Question Selection Pipeline.")
    return pipeline

def create_onboarding_data_extraction_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to extract structured data (name, email, company)
    from a user's response using an LLM.
    """
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_generator = OpenAIGenerator(api_key=openai_api_key, model="gpt-3.5-turbo")
        logger.info("OpenAI Generator instance created for onboarding data extraction.")
    except ValueError:
        logger.error("OPENAI_API_KEY environment variable not found. Cannot create LLM component.")
        return None
    if not llm_generator:
        logger.error("LLM Generator not available. Cannot create onboarding data extraction pipeline.")
        return None

    pipeline = Pipeline()

    prompt_template = """
    You are an assistant helping to extract data from user responses during an onboarding to a maternal health chatbot service.
    The user has already provided the following information:
    User Context:
    {% for key, value in user_context.items() %}
    - {{ key }}: {{ value }}
    {% endfor %}

    Here is their current chat history:
    Chat History:
    {% for message in chat_history %}
    - {{ message }}
    {% endfor %}

    Analyze the next User Message below and extract the following information if it is present:
    Common data points and their correctly formatted possible values (these data points may never assume other values than the ones listed):
    - province ["Eastern Cape","Free State","Gauteng","KwaZulu-Natal","Limpopo","Mpumalanga","Northern Cape","North West","Western Cape"]
    - area_type ["City","Township or suburb","Town","Farm or smallholding","Village","Rural area"]
    - relationship_status ["Single","Relationship","Married","Skip"]
    - education_level ["No school","Some primary","Finished primary","Some high school","Finished high school","More than high school","Don’t know","Skip"]
    - hunger_days ["0 days","1-2 days","3-4 days","5-7 days"]
    - num_children ["0","1","2","3","More than 3","Why do you ask?"]
    - phone_ownership ["Yes","No","Skip"]
    Any other observed data points that may be valuable in the context of maternal health and the current user e.g.:
    - notification_timing_preference ["Morning","Afternoon","Evening"]
    - communication_frequency_preference ["Daily","Weekly","Monthly"]
    - health_concerns [...]
    - ... (Feel free to add more as needed. The are no predefined values for these, as the user may provide any extra information.)

    Note that the user may provide ambiguous or incomplete information, but the Chat History should help you.
    Return the extracted data as a JSON object of key-value pairs corresponding to all the extracted data.
    Ensure the output is ONLY the JSON object and nothing else.
    If the response contains an answer to the last common data point that the was requested from the user, but the response is not clear enough to determine which of the possible values it should map to, then do not include it in the output).

    User Message:
    {{ user_response }}

    JSON Output:
    """
    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["user_response", "user_context", "chat_history"],
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    logger.info("Created Onboarding Data Extraction Pipeline.")
    return pipeline

def create_assessment_contextualization_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to fetch an assessment question based on flow_id and question_number,
    then contextualize it slightly using an LLM based on user context.
    """
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_generator = OpenAIGenerator(api_key=openai_api_key, model="gpt-3.5-turbo")
        logger.info("OpenAI Generator instance created for assessment contextualization.")
    except ValueError:
        logger.error("OPENAI_API_KEY environment variable not found. Cannot create LLM component.")
        return None
    if not llm_generator:
        logger.error("LLM Generator not available. Cannot create assessment contextualization pipeline.")
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
    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["user_context", "documents"]
    )

    retriever = FilterRetriever(document_store=document_store)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    logger.info("Created Assessment Contextualization Pipeline.")
    return pipeline


def create_assessment_response_validator_pipeline() -> Pipeline | None:
    """
    Creates a pipeline to validate the user's response to an assessment question.
    """
    try:
        openai_api_key = Secret.from_env_var("OPENAI_API_KEY")
        llm_generator = OpenAIGenerator(api_key=openai_api_key, model="gpt-3.5-turbo")
        logger.info("OpenAI Generator instance created for assessment response validation.")
    except ValueError:
        logger.error("OPENAI_API_KEY environment variable not found. Cannot create LLM component.")
        return None
    if not llm_generator:
        logger.error("LLM Generator not available. Cannot create assessment response validation pipeline.")
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
    prompt_builder = PromptBuilder(
        template=prompt_template,
        required_variables=["user_response"]
    )

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)

    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    logger.info("Created Assessment Response Validation Pipeline.")
    return pipeline
