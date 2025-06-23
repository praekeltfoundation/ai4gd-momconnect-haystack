import logging
from os import environ
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.weaviate import (
    AuthApiKey,
    WeaviateDocumentStore,
)
from weaviate.embedded import EmbeddedOptions

from .utilities import read_json

# --- Configurations ---
load_dotenv()

logger = logging.getLogger(__name__)

embedding_model_name = "text-embedding-3-small"


# --- Data ---
# Onboarding Flows Data
data_dir = Path("src/ai4gd_momconnect_haystack/static_content")
ONBOARDING_FLOW = read_json(data_dir / "onboarding.json")

# Assessment Questions Data
DMA_FLOW = read_json(data_dir / "dma.json")
KAB_FLOW = read_json(data_dir / "kab.json")

# ANC Follow-Up Questions Data
ANC_SURVEY_FLOW = read_json(data_dir / "anc_survey.json")

# FAQ Data
FAQ_DATA = read_json(data_dir / "faqs.json")


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
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store))
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    logger.info("Created document embedding and writing pipeline.")
    return indexing_pipeline


def ingest_content(
    indexing_pipeline: Pipeline, content_flows: dict[str, list[dict[str, Any]]]
):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content_flows: A dictionary where keys are flow_ids and values are lists
                       of content dictionaries (like ONBOARDING_FLOW or DMA_FLOW).
    """
    documents_to_ingest: list[Document] = []
    flow_count = 0
    doc_count = 0

    for flow_id, content_pieces in content_flows.items():
        print("INGESTING", flow_id)
        flow_count += 1
        for piece in content_pieces:
            doc_count += 1
            if flow_id == "anc-survey":
                doc = Document(
                    content=piece["content"],
                    meta={
                        "flow_id": flow_id,
                        "title": piece["title"],
                        "content_type": piece["content_type"],
                        "valid_responses": piece.get("valid_responses", []),
                    },
                )
            elif flow_id == "faqs":
                doc = Document(
                    content=piece["content"],
                    meta={
                        "flow_id": flow_id,
                        "title": piece["title"],
                        "valid_responses": piece.get("valid_responses", []),
                    },
                )
            elif flow_id == "onboarding":
                doc = Document(
                    content=piece["content"],
                    meta={
                        "flow_id": flow_id,
                        "question_number": piece["question_number"],
                        "content_type": piece["content_type"],
                        "valid_responses": piece.get("valid_responses", []),
                    },
                )
            else:
                # For DMA and KAB flows
                doc = Document(
                    content=piece["content"],
                    meta={
                        "flow_id": flow_id,
                        "question_number": piece["question_number"],
                        "content_type": piece["content_type"],
                        "valid_responses": [
                            item["response"]
                            for item in piece.get("valid_responses_and_scores", [])
                        ],
                    },
                )
            documents_to_ingest.append(doc)

    if documents_to_ingest:
        logger.info(
            f"Starting ingestion of {doc_count} documents from {flow_count} flows..."
        )
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")


def get_remaining_onboarding_questions(
    user_context: dict[str, Any], all_onboarding_questions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
        if question_data.get("collects") not in collected_data_points:
            remaining_questions.append(question_data)
    return remaining_questions


# --- Setup ---
def setup_document_store() -> WeaviateDocumentStore:
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
    if initial_doc_count == 0:
        logger.info("Document store is empty. Proceeding with ingestion.")
        indexing_pipe = create_embedding_pipeline(document_store)
        logger.info("--- Ingesting Onboarding Content ---")
        ingest_content(indexing_pipe, ONBOARDING_FLOW)
        logger.info("--- Ingesting Assessment Content ---")
        ingest_content(indexing_pipe, DMA_FLOW)
        ingest_content(indexing_pipe, KAB_FLOW)
        logger.info("--- Ingesting ANC Survey Content ---")
        ingest_content(indexing_pipe, ANC_SURVEY_FLOW)
        logger.info("--- Ingesting FAQ Content ---")
        ingest_content(indexing_pipe, FAQ_DATA)
    else:
        logger.info(
            "Documents found in the store. Skipping ingestion to avoid duplicates."
        )

    final_doc_count = document_store.count_documents()
    logger.info(f"Total documents in store after setup: {final_doc_count}")

    return document_store
