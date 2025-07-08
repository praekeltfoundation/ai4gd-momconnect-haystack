NEXT_ONBOARDING_QUESTION_PROMPT = """
Your task is to select the single remaining question from the list below that would be the most natural and effective to ask next and contextualize it if you think it's needed.
- You can reference the existing chat history, or the user context below to modify the tonality and/or phrasing for the user, but DO NOT change the core meaning of the question or introduce ambiguity. If no changes are needed, return the original question text.
- DO NOT list the valid responses in your contextualized version of the chosen question.
- Ensure that a dialogue using your contextualized version of the question flows smoothly (e.g. the first message in a chat must not start as if there were preceding messages), and that the contextualized question still allows for its corresponding valid responses to be grammatically valid and natural responses.

You MUST respond with a valid JSON object containing exactly these fields:
- "chosen_question_number" (integer): The question_number of the chosen question from the list below.
- "contextualized_question" (string): Your contextualized question.

User Context:
{% for key, value in user_context.items() %}
- "{{ key }}": "{{ value }}"
{% endfor %}

Remaining questions to complete user profile:
{% for q in remaining_questions %}
Question {{ q.question_number }}: "{{ q.content }}" (with valid possible responses: "{{ q.valid_responses }}")
{% endfor %}

JSON Response:
"""


ONBOARDING_DATA_EXTRACTION_PROMPT = """
You are an AI assistant collecting data during onboarding on a maternal health chatbot. Consider the chat history up to now, and the user context and latest user response below.

User Context:
{% for key, value in user_context.items() %}
- {{ key }}: {{ value }}
{% endfor %}

User's latest message:
"{{ user_response }}"

Your task is to use the 'extract_onboarding_data' tool to analyze the user's latest message in light of the chat history and extract data from their intended response:
- The extracted data **MUST** adhere strictly to the corresponding property's enums. If the user's response for one of the properties does **NOT** contain a word, phrase or index that clearly and unambiguously maps to one of the enums, DO NOT include that property in your tool call. Only store the 'Skip' enum value for these properties if the user explicitly states they want to skip.
- Only include a field if you are highly confident that the user's input maps to an allowed 'enum' value.
- Do not extract a data point if it clearly has already been collected in the user context, unless the user's latest message explicitly provides new information that updates it.
- For properties with numeric ranges like 'hunger_days', you MUST map the user's input to the correct enum category whose range contains or corresponds to the user's input, unless they did not provide valid information. Do not just look for an exact string match. As examples:
    - If the user says "3", you should extract: {"hunger_days": "3-4 days"}
    - If the user says "one day", you should extract: {"hunger_days": "1-2 days"}
    - If the user says "6", you should extract: {"hunger_days": "5-7 days"}
    - If the user says "I haven't been hungry", you should extract: {"hunger_days": "0 days"}
- For 'num_children', apply similar numeric mapping logic. If the user indicates they have any number of children greater than 3 (e.g. 4, 5, 6...), you should extract: {"num_children": "More than 3"}
- Regarding the 'education_level', note that grades 1-7 correspond to primary school, while grades 8-12 correspond to high school.
- For the open-ended additionalProperties, extract any extra information mentioned that is not already in one of the expected properties, and AS LONG AS it pertains specifically to maternal health or the use of a maternal health chatbot.
    """

ASSESSMENT_CONTEXTUALIZATION_PROMPT = """
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

Review the Original Assessment Question above. If you think it's needed, make minor adjustments to ensure that the question is clear and directly applicable to the user's context. **Crucially, do not change the core meaning, difficulty, the scale/format, or the applicability to the valid responses, of the question.**

Before finalizing, silently check if a user's response of "{{ documents[0].meta.valid_responses[0] }}" would be a grammatically correct and natural answer to your generated question. The question must be a complete, well-formed sentence.

If no changes are needed, return the original question text.

You MUST respond with a valid JSON object containing exactly one key: "contextualized_question".

JSON Response:
    """

ASSESSMENT_RESPONSE_VALIDATOR_PROMPT = """
You are an AI assistant validating a user's response to an assessment question in a chatbot for new mothers in South Africa.
Your task is to analyze the user's response and determine if it maps to one of the allowed responses provided below.

Allowed Responses:
{{ valid_responses_for_prompt }}

User Response:
"{{ user_response }}"

You MUST respond with a valid JSON object. The JSON should contain a single key, "validated_response".

- If the user's response clearly and unambiguously corresponds to one of the "Allowed Responses" (or a numerical/alphabetical index of an "Allowed Response", if there are indices present in the list above), the value of "validated_response" should be the exact text of that corresponding allowed response.
- If the user's response is ambiguous, does not map to any of the allowed responses, or is nonsense/gibberish, you MUST set the value of "validated_response" to "nonsense".

JSON Response:
    """

ASSESSMENT_END_RESPONSE_VALIDATOR_PROMPT = """
You are an AI assistant validating a user's response to the previous message sent by a chatbot for new mothers in South Africa.
Your task is to analyze the user's response and determine if it maps to one of the allowed responses provided below.

Previous Message:
"{{ previous_message }}"

User Response:
"{{ user_response }}"

You MUST respond with a valid JSON object. The JSON should contain a single key, "validated_response".

- If the user's response clearly and unambiguously corresponds to one of the expected responses of the previous message, the value of "validated_response" should be the exact text of that expected response.
- If the user's response is ambiguous, does not match any of the expected responses, or is nonsense/gibberish, you MUST set the value of "validated_response" to "nonsense".

JSON Response:
    """

ANC_SURVEY_CONTEXTUALIZATION_PROMPT = """
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

CLINIC_VISIT_DATA_EXTRACTION_PROMPT = """
You are an AI assistant helping to extract information from a user's (a new South African mother) response to a pregnancy clinic visit survey question/message into a structured format.

Your task is to analyze the user's response in light of the previous survey question/message and its expected responses. You must determine which of the expected responses the user's response maps to in meaning and intent.

Follow these rules:
1.  Map responses based on meaning and intent, not just exact string matching. This includes slang, colloquialisms, synonyms, and shortened versions.
2.  The value for "validated_response" MUST be the exact text of the matched expected response.
3.  If the options are presented in a lettered or numbered list (e.g., 'a.', '1.'), you MUST strip this prefix from your response. The `validated_response` should contain ONLY the text of the option itself.
4.  You MUST respond with a valid JSON object with a single key, "validated_response".

Here are some examples of how to perform this task:

---
**Example 1:**

Previous survey question/message:
"That's great news! 🌟 We'd love to hear about your check-up. Do you have a couple of minutes to share with us?

Please reply with one of the following:
- 'Yes'
- 'Remind me tomorrow'"

User's latest response:
- "Ok"

JSON Response:
{
    "validated_response": "Yes"
}
---
**Example 2:**

Previous survey question/message:
"Did you manage to get to the clinic for your check-up? Please reply 'Yes, I went' or 'No, not yet'"

User's latest response:
- "i went"

JSON Response:
{
    "validated_response": "Yes, I went"
}
---
**Example 3:**

Previous survey question/message:
"Will you go to your next check-up? Please reply 'Yes, I will' or 'No, I won't'"

User's latest response:
- "yes"

JSON Response:
{
    "validated_response": "Yes, I will"
}
---
**Example 4:**

Previous survey question/message:
"How are you feeling today? You can reply with 'Good 😊', 'Okay 😐', or 'Something else 😞'"

User's latest response:
- "other"

JSON Response:
{
    "validated_response": "Something else 😞"
}
---
**Example 5:**

Previous survey question/message:
"That's great news! 🌟 We'd love to hear about your check-up. Do you have a couple of minutes to share with us?

Please reply with one of the following:
- 'Yes'
- 'Remind me tomorrow'"

User's latest response:
- "asdfghjkl"

JSON Response:
{
    "validated_response": "nonsense"
}
---
**Example 6 (Handling Lettered Lists):**


Previous survey question/message:
"Overall, how easy was it for you to provide your feedback?
a. Very easy
b. A little easy
c. OK"

User's latest response:
- "a"
JSON Response:
{
 "validated_response": "Very easy"
}
---

**Now, perform the same task for the following new input:**

Previous survey question/message:
"{{ previous_service_message }}"

User's latest response:
- "{{ user_response }}"

JSON Response:
    """

INTENT_DETECTION_PROMPT = """
You are an AI assitant performing intent classification of user responses in a maternal health chatbot.

Last question that was sent to the user:
"{{ last_question }}"

User's response:
"{{ user_response }}"

Please classify the user's response, in light of the last question sent to the user, into one of these intents:
- 'JOURNEY_RESPONSE': The user is directly answering, attempting to answer, or skipping the question asked.
- 'QUESTION_ABOUT_STUDY': The user is asking a question about the research study itself (e.g., "who are you?", "why are you asking this?").
- 'HEALTH_QUESTION': The user is asking a new question related to health, pregnancy, or their wellbeing, instead of answering the question.
- 'ASKING_TO_STOP_MESSAGES': The user explicitly expresses a desire to stop receiving messages.
- 'ASKING_TO_DELETE_DATA': The user explicitly expresses a desire to leave the study and have their data deleted.
- 'REPORTING_AIRTIME_NOT_RECEIVED': The user is reporting that they have not received their airtime incentive.
- 'CHITCHAT': The user is making a conversational comment that is not an answer, question or request.

You MUST respond with a valid JSON object containing exactly one key: "intent".

JSON Response:
    """

FAQ_PROMPT = """
You are a helpful assistant for a maternal health chatbot. Answer the user's question based ONLY on the provided context information.
If the context does not contain the answer, say that you do not have that information.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

User's Question: {{ user_question }}

Answer:
    """
