from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
from string import ascii_lowercase

from haystack.dataclasses import ChatMessage
from pydantic import BaseModel, ValidationError

from ai4gd_momconnect_haystack import doc_store
from ai4gd_momconnect_haystack.enums import AssessmentType
from ai4gd_momconnect_haystack.pydantic_models import AssessmentQuestion


logger = logging.getLogger(__name__)

onboarding_flow_id = "onboarding"
dma_pre_flow_id = "dma-pre-assessment"
dma_post_flow_id = "dma-post-assessment"
kab_k_pre_flow_id = "knowledge-pre-assessment"
kab_k_post_flow_id = "knowledge-post-assessment"
kab_a_pre_flow_id = "attitude-pre-assessment"
kab_a_post_flow_id = "attitude-post-assessment"
kab_b_pre_flow_id = "behaviour-pre-assessment"
kab_b_post_flow_id = "behaviour-post-assessment"
anc_survey_flow_id = "anc-survey"
faqs_flow_id = "faqs"

all_onboarding_questions = doc_store.ONBOARDING_FLOW
all_anc_survey_questions = doc_store.ANC_SURVEY_FLOW
faq_questions = doc_store.FAQ_DATA
# The following assessments have the same pre and post flows, so we can use the same IDs
all_dma_questions = doc_store.DMA_FLOW
all_kab_k_questions = doc_store.KAB_K_FLOW
all_kab_a_questions = doc_store.KAB_A_FLOW
# The kab_b pre and post flows are different, so we use separate IDs
all_kab_b_pre_questions = doc_store.KAB_B_PRE_FLOW
all_kab_b_post_questions = doc_store.KAB_B_POST_FLOW
# We also need content for messaging that happens at the end of assessments:
dma_end_messaging_pre = doc_store.DMA_ASSESSMENT_END_FLOW
kab_k_end_messaging_pre = doc_store.KAB_K_ASSESSMENT_END_FLOW
kab_a_end_messaging_pre = doc_store.KAB_A_ASSESSMENT_END_FLOW
kab_b_end_messaging_pre = doc_store.KAB_B_ASSESSMENT_END_FLOW

assessment_flow_map = {
    dma_pre_flow_id: all_dma_questions,
    dma_post_flow_id: all_dma_questions,
    kab_k_pre_flow_id: all_kab_k_questions,
    kab_k_post_flow_id: all_kab_k_questions,
    kab_a_pre_flow_id: all_kab_a_questions,
    kab_a_post_flow_id: all_kab_a_questions,
    kab_b_pre_flow_id: all_kab_b_pre_questions,
    kab_b_post_flow_id: all_kab_b_post_questions,
}

FLOWS_WITH_INTRO = [
    onboarding_flow_id,
    kab_b_pre_flow_id,
    kab_b_post_flow_id,
    anc_survey_flow_id,
]

assessment_end_flow_map = {
    dma_pre_flow_id: dma_end_messaging_pre,
    kab_k_pre_flow_id: kab_k_end_messaging_pre,
    kab_a_pre_flow_id: kab_a_end_messaging_pre,
    kab_b_pre_flow_id: kab_b_end_messaging_pre,
}

assessment_map_to_their_pre = {
    dma_pre_flow_id: AssessmentType.dma_pre_assessment,
    dma_post_flow_id: AssessmentType.dma_pre_assessment,
    kab_k_pre_flow_id: AssessmentType.knowledge_pre_assessment,
    kab_k_post_flow_id: AssessmentType.knowledge_pre_assessment,
    kab_a_pre_flow_id: AssessmentType.attitude_pre_assessment,
    kab_a_post_flow_id: AssessmentType.attitude_pre_assessment,
    kab_b_pre_flow_id: AssessmentType.behaviour_pre_assessment,
    kab_b_post_flow_id: AssessmentType.behaviour_pre_assessment,
}

assessment_map_to_assessment_types = {
    dma_pre_flow_id: AssessmentType.dma_pre_assessment,
    dma_post_flow_id: AssessmentType.dma_post_assessment,
    kab_k_pre_flow_id: AssessmentType.knowledge_pre_assessment,
    kab_k_post_flow_id: AssessmentType.knowledge_post_assessment,
    kab_a_pre_flow_id: AssessmentType.attitude_pre_assessment,
    kab_a_post_flow_id: AssessmentType.attitude_post_assessment,
    kab_b_pre_flow_id: AssessmentType.behaviour_pre_assessment,
    kab_b_post_flow_id: AssessmentType.behaviour_post_assessment,
}

ANC_SURVEY_MAP = {item.title: item for item in all_anc_survey_questions}


def generate_scenario_id(flow_type: str, username: str) -> str:
    """Generates a unique scenario ID."""
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return f"{flow_type}_{username}_{timestamp}"


def load_json_and_validate(
    file_path: Path, model: type[BaseModel] | type[dict]
) -> Any | None:
    """
    Loads a JSON file and validates its content against a Pydantic model or as a dict.
    This is the primary gateway for safely loading any external JSON data.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Guard Clause: If the model is just 'dict', we are loading a raw
        # doc_store. Return it directly without Pydantic validation.
        # The validation for its contents is handled later in tasks.py.
        if model is dict:
            return raw_data

        # If the model is a Pydantic model, proceed with validation.
        if issubclass(model, BaseModel):
            if isinstance(raw_data, list):
                return [model.model_validate(item) for item in raw_data]
            return model.model_validate(raw_data)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except ValidationError as e:
        logging.error(f"Data validation error in {file_path}:\n{e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred with {file_path}: {e}")
    return None


def save_json_file(data: list[dict[str, Any]], file_path: Path) -> None:
    """Saves the final processed data to a JSON file."""
    try:
        # Ensure the output directory exists before writing.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Successfully saved final augmented output to {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing to {file_path}: {e}")


def chat_messages_to_json(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Converts a list of ChatMessage objects to a JSON-serializable list of dicts."""
    return [
        {
            "role": msg.role.value,
            "text": msg.text,
            "meta": {k: v for k, v in msg.meta.items()},
        }
        for msg in messages
    ]


def prepend_valid_responses_with_alphabetical_index(
    valid_responses: list[str],
) -> list[str]:
    """
    Prepends an alphabetical index (a., b., c., ...) to each string in a list.
    """
    return [
        f"{letter}. {response}"
        for letter, response in zip(ascii_lowercase, valid_responses)
    ]


def prepend_valid_responses_with_alphabetical_index_and_responses_in_quotes(
    valid_responses: list[str],
) -> list[str]:
    """
    Prepends an alphabetical index (a., b., c., ...) to each string in a list.
    """
    return [
        f'{letter}. "{response}"'
        for letter, response in zip(ascii_lowercase, valid_responses)
    ]


def get_likert_scale_with_numeric_index() -> list[str]:
    """
    Returns a hardcoded numerically-indexed Likert scale.
    """
    return [
        "1. I strongly agree",
        "2. I agree",
        "3. I'm not sure",
        "4. I disagree",
        "5. I strongly disagree",
    ]


def get_likert_scale_with_numeric_index_and_responses_in_quotes() -> list[str]:
    """
    Returns a hardcoded numerically-indexed Likert scale.
    """
    return [
        '1. "I strongly agree"',
        '2. "I agree"',
        '3. "I\'m not sure"',
        '4. "I disagree"',
        '5. "I strongly disagree"',
    ]


def prepare_valid_responses_to_display_to_anc_survey_user(
    text_to_prepend: str, question: str, valid_responses: list[str], step_title: str
) -> str:
    final_question_text = text_to_prepend + question
    if valid_responses:
        if step_title in ["start", "seen_yes", "Q_seen_no", "start_not_going"]:
            options = "\n\n" + "\n".join(
                ["Please reply with one of the following:"]
                + [f"- '{vr}'" for vr in valid_responses]
            )
            final_question_text += options
        elif step_title in ["bad", "good"]:
            vr = valid_responses[-1]
            final_question_text += f"\n\nPlease reply with '{vr}' to continue."
        elif step_title in ["Q_experience", "feedback_if_first_survey"]:
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(valid_responses)
            )
            final_question_text += options
        elif step_title == "Q_visit_bad":
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        "I didn't have my maternity record📝",
                        "I was shamed or embarrassed 😳",
                        "I was not given privacy to discuss my worries or challenges 🤐",
                        "I was not given enough information about tests, supplements or my pregnancy ℹ️",
                        "The staff were disrespectful 🤬",
                        "They asked me to pay 💰",
                        "I had to wait a long time ⌛",
                        "Something else 😞",
                    ]
                )
            )
            final_question_text += options
        elif step_title == "Q_visit_good":
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        "No problems - all fine👌",
                        "I didn't have my maternity record📝",
                        "I was shamed or embarrassed 😳",
                        "I was not given privacy to discuss my worries or challenges 🤐",
                        "I was not given enough information about tests, supplements or my pregnancy ℹ️",
                        "The staff were disrespectful 🤬",
                        "They asked me to pay 💰",
                        "I had to wait a long time ⌛",
                        "Something else 😞",
                    ]
                )
            )
            final_question_text += options
        elif step_title == "Q_challenges":
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        "No challenges - all fine👌",
                        "Transport is expensive or it’s far to travel 🚌",
                        "I don't have support from my partner or family 🤝",
                        "It's hard to get there during clinic opening hours 🏥",
                        "Something else 😞",
                    ]
                )
            )
            final_question_text += options
        elif step_title == "Q_why_no_visit":
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        "The clinic was closed ⛔",
                        "I had to wait too long⌛",
                        "I didn't have my maternity record 📝",
                        "They asked me to pay💰",
                        "I was told to come back another day 📅",
                        "I left because the staff were disrespectful 🤬",
                        "Something else 😞",
                    ]
                )
            )
            final_question_text += options
        elif step_title == "Q_why_not_go":
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        "I didn't know I had a check-up 📅",
                        "I didn't know where to go📍",
                        "I don't want check-ups ⛔",
                        "I can't go in clinic opening hours 🏥",
                        "They asked me to pay💰",
                        "Waiting times are too long ⌛",
                        "I have no support from family or friends 🤝",
                        "Getting to the clinic is hard - no money for transport or it's too far to travel 🚌",
                        "I forgot about it 😧",
                        "Something else 😞",
                    ]
                )
            )
            final_question_text += options
    return final_question_text


def prepare_valid_responses_to_use_in_anc_survey_system_prompt(
    question: str, valid_responses: list[str], step_title: str
) -> str:
    final_question_text = question
    if valid_responses:
        if step_title in ["start", "seen_yes", "Q_seen_no", "start_not_going"]:
            options = "\n\n" + "\n".join(
                ["Please reply with one of the following:"]
                + [f"- '{vr}'" for vr in valid_responses]
            )
            final_question_text += options
        elif step_title in ["bad", "good"]:
            vr = valid_responses[-1]
            final_question_text += f"\n\nPlease reply with '{vr}' to continue."
        elif step_title in [
            "Q_experience",
            "Q_visit_bad",
            "Q_visit_good",
            "Q_challenges",
            "Q_why_no_visit",
            "Q_why_not_go",
            "feedback_if_first_survey",
        ]:
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(valid_responses)
            )
            final_question_text += options
        else:
            options = "\n\n" + "\n".join([f"- '{vr}'" for vr in valid_responses])
            final_question_text += options

    return final_question_text


def prepare_valid_responses_to_display_to_assessment_user(
    flow_id: str, question_number: int, question: str, question_data: AssessmentQuestion
) -> str:
    if question_data.valid_responses_and_scores:
        if "dma" in flow_id or "attitude" in flow_id:
            options = "\n\n" + "\n".join(
                prepend_valid_responses_with_alphabetical_index(
                    [
                        item.response
                        for item in question_data.valid_responses_and_scores
                        if item.response != "Skip"
                    ]
                )
            )
            question = f"{question}{options}"
        elif flow_id == "behaviour-pre-assessment":
            if question_number == 3:
                question = question + "\n\n0 or 1?"
            if question_number == 4:
                options = "\n\n" + "\n".join(["0 or 1?"])
                question = question + '\n\nPlease reply with "Yes" or "No".'
        else:
            if question_number in [1, 3, 5]:
                options = "\n\n" + "\n".join(
                    prepend_valid_responses_with_alphabetical_index(
                        [
                            item.response
                            for item in question_data.valid_responses_and_scores
                            if item.response != "Skip"
                        ]
                    )
                )
                question = f"{question}{options}"
            if question_number == 2:
                question = f"{question}\n\na. You have contractions that stop when you move\nb. Your tummy stays tight, hard and is very painful\nc. Your lower back aches all day"
    return question


def prepare_valid_responses_to_use_in_assessment_system_prompt(
    flow_id: str, question_number: int, question_data: AssessmentQuestion
) -> str:
    if question_data.valid_responses_and_scores:
        if flow_id in ["dma-assessment", "attitude-assessment"]:
            options = "\n".join(
                get_likert_scale_with_numeric_index_and_responses_in_quotes()
                + ['"Skip"']
            )
        elif flow_id == "behaviour-pre-assessment":
            if question_number == 3:
                options = "\n".join(['- "0"', '- "1"', '- "Skip"'])
            elif question_number == 4:
                options = "\n".join(['- "Yes"', '- "No"', '- "Skip"'])
            else:
                options = "\n".join(
                    [
                        f'- "{vr}"'
                        for vr in [
                            item.response
                            for item in question_data.valid_responses_and_scores
                        ]
                    ]
                )
        else:
            if question_number in [1, 3, 5]:
                options = "\n".join(
                    prepend_valid_responses_with_alphabetical_index_and_responses_in_quotes(
                        [
                            item.response
                            for item in question_data.valid_responses_and_scores
                            if item.response != "Skip"
                        ]
                    )
                    + ['"Skip"']
                )
            elif question_number == 2:
                options = "a. Contractions stop\nb. Tight painful tummy\nc. Lower back ache\nI don't know\nSkip"
            else:
                options = "\n".join(
                    [
                        f'- "{vr}"'
                        for vr in [
                            item.response
                            for item in question_data.valid_responses_and_scores
                        ]
                    ]
                )
    return options


def prepare_valid_responses_to_display_to_onboarding_user(
    question: str, collects: str, valid_responses: list[str]
) -> str:
    valid_responses.remove("Skip") if "Skip" in valid_responses else None
    final_question_text = question
    if valid_responses:
        if collects in [
            "province",
            "area_type",
            "relationship_status",
            "education_level",
        ]:
            valid_responses = prepend_valid_responses_with_alphabetical_index(
                valid_responses
            )
            options = "\n\n" + "\n".join(valid_responses)
            final_question_text += options
    return final_question_text
