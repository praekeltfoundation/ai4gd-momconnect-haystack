import json
import logging
from os import environ
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.weaviate import (
    AuthApiKey,
    WeaviateDocumentStore,
)
from pydantic import TypeAdapter, ValidationError
from weaviate.embedded import EmbeddedOptions

from ai4gd_momconnect_haystack.pydantic_models import (
    FAQ,
    ANCSurveyQuestion,
    AssessmentEndItem,
    AssessmentQuestion,
    OnboardingQuestion,
)

# --- Configurations ---
load_dotenv()

logger = logging.getLogger(__name__)

embedding_model_name = "text-embedding-3-small"


# --- Data ---
def load_content_json_and_validate(
    file_path: Path, model: object, flow_id: str
) -> list | None:
    """
    Loads a JSON file and validates its content against a Pydantic model.
    This is the primary gateway for safely loading any external JSON data.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items_to_validate = raw_data.get(flow_id)
        if items_to_validate is None:
            logging.error(f"Flow ID '{flow_id}' not found in {file_path}")
            return None

        if not isinstance(items_to_validate, list):
            logging.error(
                f"Expected a list for flow_id '{flow_id}' in {file_path}, "
                f"but got {type(items_to_validate).__name__}"
            )
            return None

        list_adapter = TypeAdapter(list[model])  # type: ignore[valid-type]

        return list_adapter.validate_python(items_to_validate)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except ValidationError as e:
        logging.error(
            f"Data validation error in {file_path} for flow '{flow_id}':\n{e}"
        )
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred with {file_path}: {e}")

    return None


data_dir = Path("src/ai4gd_momconnect_haystack/static_content")

# Onboarding Flows Data
onboarding_flow = load_content_json_and_validate(
    data_dir / "onboarding.json", OnboardingQuestion, "onboarding"
)
assert onboarding_flow is not None, "Failed to load the onboarding flow."
ONBOARDING_FLOW: list[OnboardingQuestion] = onboarding_flow

# Assessment Questions Data
dma_flow = load_content_json_and_validate(
    data_dir / "dma.json", AssessmentQuestion, "dma-assessment"
)
assert dma_flow is not None, "Failed to load the DMA flow."
DMA_FLOW: list[AssessmentQuestion] = dma_flow

kab_k_flow = load_content_json_and_validate(
    data_dir / "kab.json", AssessmentQuestion, "knowledge-assessment"
)
assert kab_k_flow is not None, "Failed to load the KAB Knowledge flow."
KAB_K_FLOW: list[AssessmentQuestion] = kab_k_flow

kab_a_flow = load_content_json_and_validate(
    data_dir / "kab.json", AssessmentQuestion, "attitude-assessment"
)
assert kab_a_flow is not None, "Failed to load the KAB Attitude flow."
KAB_A_FLOW: list[AssessmentQuestion] = kab_a_flow

kab_b_pre_flow = load_content_json_and_validate(
    data_dir / "kab.json", AssessmentQuestion, "behaviour-pre-assessment"
)
assert kab_b_pre_flow is not None, "Failed to load the KAB Behaviour (pre) flow."
KAB_B_PRE_FLOW: list[AssessmentQuestion] = kab_b_pre_flow

kab_b_post_flow = load_content_json_and_validate(
    data_dir / "kab.json", AssessmentQuestion, "behaviour-post-assessment"
)
assert kab_b_post_flow is not None, "Failed to load the KAB Behaviour (post) flow."
KAB_B_POST_FLOW: list[AssessmentQuestion] = kab_b_post_flow

# Assessment End Messaging Data
dma_assessment_end_flow: list[AssessmentEndItem] | None = (
    load_content_json_and_validate(
        data_dir / "assessment_ends.json", AssessmentEndItem, "dma-pre-assessment"
    )
)
assert dma_assessment_end_flow is not None, (
    "Failed to load the DMA assessment end flow."
)
DMA_ASSESSMENT_END_FLOW: list[AssessmentEndItem] = dma_assessment_end_flow

kab_k_assessment_end_flow: list[AssessmentEndItem] | None = (
    load_content_json_and_validate(
        data_dir / "assessment_ends.json", AssessmentEndItem, "knowledge-pre-assessment"
    )
)
assert kab_k_assessment_end_flow is not None, (
    "Failed to load the KAB Knowledge assessment end flow."
)
KAB_K_ASSESSMENT_END_FLOW: list[AssessmentEndItem] = kab_k_assessment_end_flow

kab_a_assessment_end_flow: list[AssessmentEndItem] | None = (
    load_content_json_and_validate(
        data_dir / "assessment_ends.json", AssessmentEndItem, "attitude-pre-assessment"
    )
)
assert kab_a_assessment_end_flow is not None, (
    "Failed to load the KAB Attitude assessment end flow."
)
KAB_A_ASSESSMENT_END_FLOW: list[AssessmentEndItem] = kab_a_assessment_end_flow

kab_b_assessment_end_flow: list[AssessmentEndItem] | None = (
    load_content_json_and_validate(
        data_dir / "assessment_ends.json", AssessmentEndItem, "behaviour-pre-assessment"
    )
)
assert kab_b_assessment_end_flow is not None, (
    "Failed to load the KAB Behaviour assessment end flow."
)
KAB_B_ASSESSMENT_END_FLOW: list[AssessmentEndItem] = kab_b_assessment_end_flow

# ANC Follow-Up Questions Data
anc_survey_flow = load_content_json_and_validate(
    data_dir / "anc_survey.json", ANCSurveyQuestion, "anc-survey"
)
assert anc_survey_flow is not None, "Failed to load the ANC survey flow."
ANC_SURVEY_FLOW: list[ANCSurveyQuestion] = anc_survey_flow

# FAQ Data
faq_data = load_content_json_and_validate(data_dir / "faqs.json", FAQ, "faqs")
assert faq_data is not None, "Failed to load the FAQs."
FAQ_DATA: list[FAQ] = faq_data


# --- Core Functions ---
def initialize_document_store() -> WeaviateDocumentStore:
    """
    Initializes the WeaviateDocumentStore with the specified path and settings.

    Returns:
        An instance of WeaviateDocumentStore.
    """
    if weaviate_url := environ.get("WEAVIATE_URL"):
        if environ.get("WEAVIATE_API_KEY"):
            doc_store = WeaviateDocumentStore(
                url=weaviate_url, auth_client_secret=AuthApiKey()
            )
        else:
            doc_store = WeaviateDocumentStore(url=weaviate_url)
    else:
        doc_store = WeaviateDocumentStore(embedded_options=EmbeddedOptions())
    logger.info(f"Initialized Document Store: {type(doc_store).__name__}")
    return doc_store


def create_embedding_pipeline(doc_store: WeaviateDocumentStore) -> Pipeline:
    """
    Creates a Haystack pipeline for embedding and writing documents to the store.

    Args:
        doc_store: The initialized Haystack ChromaDocumentStore.

    Returns:
        A Haystack Pipeline configured for indexing.
    """
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=embedding_model_name,
            progress_bar=True,
        ),
    )
    indexing_pipeline.add_component(
        "writer",
        DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.OVERWRITE),
    )
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    logger.info("Created document embedding and writing pipeline.")
    return indexing_pipeline


def ingest_onboarding_content(
    indexing_pipeline: Pipeline, content: list[OnboardingQuestion], flow_id: str
):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content: A list of OnboardingQuestion objects.
    """
    documents_to_ingest: list[Document] = []
    doc_count = 0

    print("INGESTING", flow_id)
    for piece in content:
        doc_count += 1
        doc = Document(
            id=f"{flow_id}-{piece.question_number}",
            content=piece.content,
            meta={
                "flow_id": flow_id,
                "question_number": piece.question_number,
                "content_type": piece.content_type,
                "valid_responses": piece.valid_responses,
            },
        )
        documents_to_ingest.append(doc)

    if documents_to_ingest:
        logger.info(f"Starting ingestion of {doc_count} documents...")
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")


def ingest_assessment_content(
    indexing_pipeline: Pipeline, content: list[AssessmentQuestion], flow_id: str
):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content: A list of AssessmentQuestion objects.
    """
    documents_to_ingest: list[Document] = []
    doc_count = 0

    print("INGESTING", flow_id)
    for piece in content:
        doc_count += 1
        if piece.valid_responses_and_scores:
            doc = Document(
                id=f"{flow_id}-{piece.question_number}",
                content=piece.content,
                meta={
                    "flow_id": flow_id,
                    "question_number": piece.question_number,
                    "content_type": piece.content_type,
                    "valid_responses": [
                        item.response for item in piece.valid_responses_and_scores
                    ],
                },
            )
        else:
            doc = Document(
                id=f"{flow_id}-{piece.question_number}",
                content=piece.content,
                meta={
                    "flow_id": flow_id,
                    "question_number": piece.question_number,
                    "content_type": piece.content_type,
                    "valid_responses": [],
                },
            )
        documents_to_ingest.append(doc)

    if documents_to_ingest:
        logger.info(f"Starting ingestion of {doc_count} documents...")
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")


def ingest_survey_content(
    indexing_pipeline: Pipeline, content: list[ANCSurveyQuestion], flow_id: str
):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content: A list of ANCSurveyQuestion objects.
    """
    documents_to_ingest: list[Document] = []
    doc_count = 0

    print("INGESTING", flow_id)
    for piece in content:
        doc_count += 1
        doc = Document(
            id=f"{flow_id}-{piece.title}",
            content=piece.content,
            meta={
                "flow_id": flow_id,
                "title": piece.title,
                "content_type": piece.content_type,
                "valid_responses": piece.valid_responses,
            },
        )
        documents_to_ingest.append(doc)

    if documents_to_ingest:
        logger.info(f"Starting ingestion of {doc_count} documents...")
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")


def ingest_faq_content(indexing_pipeline: Pipeline, content: list[FAQ], flow_id: str):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content: A list of FAQ objects.
    """
    documents_to_ingest: list[Document] = []
    doc_count = 0

    print("INGESTING", flow_id)
    for piece in content:
        doc_count += 1
        doc = Document(
            id=f"{flow_id}-{piece.title}",
            content=piece.content,
            meta={
                "flow_id": flow_id,
                "title": piece.title,
            },
        )
        documents_to_ingest.append(doc)

    if documents_to_ingest:
        logger.info(f"Starting ingestion of {doc_count} documents...")
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")


def get_remaining_onboarding_questions(
    user_context: dict[str, Any], all_onboarding_questions: list[OnboardingQuestion]
) -> list[OnboardingQuestion]:
    """
    Identifies which onboarding questions still need to be asked based on user_context.
    """
    onboarding_data_to_collect = [
        "province",
        "area_type",
        "relationship_status",
        "education_level",
        "hunger_days",
        "num_children",
        "phone_ownership",
    ]
    collected_data_points = [
        k
        for k, v in user_context.items()
        if k in onboarding_data_to_collect and v is not None
    ]
    remaining_questions = []
    for question_data in all_onboarding_questions:
        if question_data.collects not in collected_data_points:
            remaining_questions.append(question_data)
    return remaining_questions


# --- Setup ---
def setup_document_store(startup: bool = False) -> WeaviateDocumentStore:
    """
    Initializes document store and ingests data if needed.

    Returns:
        An initialized and populated WeaviateDocumentStore.
    """
    logger.info("Setting up document store...")
    document_store = initialize_document_store()

    # Check if documents already exist in the store
    initial_doc_count = document_store.count_documents()
    logger.info(f"Document store currently contains {initial_doc_count} documents.")

    # If the document store is empty, ingest the content
    if startup or initial_doc_count == 0:
        logger.info("Document store is empty. Proceeding with ingestion.")
        #if initial_doc_count > 0:
        #    all_docs = document_store.filter_documents()
        #    doc_ids = [doc.id for doc in all_docs]
        #    document_store.delete_documents(doc_ids)
        #    logger.info(f"Deleted {len(doc_ids)} old documents.")
        indexing_pipe = create_embedding_pipeline(document_store)
        logger.info("--- Ingesting Onboarding Content ---")
        ingest_onboarding_content(indexing_pipe, ONBOARDING_FLOW, "onboarding")
        logger.info("--- Ingesting Assessment Content ---")
        ingest_assessment_content(indexing_pipe, DMA_FLOW, "dma-assessment")
        ingest_assessment_content(indexing_pipe, KAB_K_FLOW, "knowledge-assessment")
        ingest_assessment_content(indexing_pipe, KAB_A_FLOW, "attitude-assessment")
        ingest_assessment_content(
            indexing_pipe, KAB_B_PRE_FLOW, "behaviour-pre-assessment"
        )
        ingest_assessment_content(
            indexing_pipe, KAB_B_POST_FLOW, "behaviour-post-assessment"
        )
        logger.info("--- Ingesting ANC Survey Content ---")
        ingest_survey_content(indexing_pipe, ANC_SURVEY_FLOW, "anc-survey")
        logger.info("--- Ingesting FAQ Content ---")
        ingest_faq_content(indexing_pipe, FAQ_DATA, "faqs")
    else:
        logger.info(
            "Documents found in the store. Skipping ingestion to avoid duplicates."
        )

    final_doc_count = document_store.count_documents()
    logger.info(f"Total documents in store after setup: {final_doc_count}")

    return document_store
