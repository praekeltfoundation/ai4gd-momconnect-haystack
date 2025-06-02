import logging
from os import environ
from dotenv import load_dotenv

from haystack import Document, Pipeline
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore, AuthApiKey
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from weaviate.embedded import EmbeddedOptions


# --- Configurations ---
load_dotenv()

logger = logging.getLogger(__name__)

embedding_model_name = "text-embedding-3-small"


# --- Data ---
# Onboarding Flows Data
ONBOARDING_FLOWS: dict[str, list[dict[str, any]]] = {
    "onboarding": [
        {
            "question_number": 1,
            "content": "Which province do you live in?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "Eastern Cape",
                "Free State",
                "Gauteng",
                "KwaZulu-Natal",
                "Limpopo",
                "Mpumalanga",
                "Northern Cape",
                "North West",
                "Western Cape"
            ],
            "collects": "province"
        },
        {
            "question_number": 2,
            "content": "And what kind of area do you live in?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "City",
                "Township or suburb",
                "Town",
                "Farm or smallholding",
                "Village",
                "Rural area"
            ],
            "collects": "area_type"
        },
        {
            "question_number": 3,
            "content": "What’s your current relationship status?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "Single",
                "Relationship",
                "Married",
                "Skip"
            ],
            "collects": "relationship_status"
        },
        {
            "question_number": 4,
            "content": "What’s your highest level of education?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "No school",
                "Some primary",
                "Finished primary",
                "Some high school",
                "Finished high school",
                "More than high school",
                "Don’t know",
                "Skip"
            ],
            "collects": "education_level"
        },
        {
            "question_number": 5,
            "content": "In the past 7 days, how many days did you not have enough to eat?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "0 days",
                "1-2 days",
                "3-4 days",
                "5-7 days"
            ],
            "collects": "hunger_days"
        },
        {
            "question_number": 6,
            "content": "How many children do you have? Count all your children of any age.",
            "content_type": "onboarding_message",
            "valid_responses": [
                "0",
                "1",
                "2",
                "3",
                "More than 3",
                "Why do you ask?"
            ],
            "collects": "num_children"
        },
        {
            "question_number": 7,
            "content": "Do you own the phone you’re using right now?",
            "content_type": "onboarding_message",
            "valid_responses": [
                "Yes",
                "No",
                "Skip"
            ],
            "collects": "phone_ownership"
        },
    ],
}

# Assessment Questions Data
ASSESSMENT_FLOWS: dict[str, list[dict[str, any]]] = {
    "dma-assessment": [
        {
            "question_number": 1,
            "content": "How confident are you in making decisions about your health?",
            "content_type": "assessment_question",
            "valid_responses": [
                "Not at all confident",
                "A little confident",
                "Somewhat confident",
                "Confident",
                "Very confident"
            ]
        },
        {
            "question_number": 2,
            "content": "How confident are you discussing your medical problems with your doctor/nurse?",
            "content_type": "assessment_question",
            "valid_responses": [
                "Not at all confident",
                "A little confident",
                "Somewhat confident",
                "Confident",
                "Very confident"
            ]
        },
        {
            "question_number": 3,
            "content": "I feel confident questioning my doctor/nurse about my treatment.",
            "content_type": "assessment_question",
            "valid_responses": [
                "Not at all confident",
                "A little confident",
                "Somewhat confident",
                "Confident",
                "Very confident"
            ]
        },
        {
            "question_number": 4,
            "content": "How confident are you in taking actions to improve your health?",
            "content_type": "assessment_question",
            "valid_responses": [
                "Not at all confident",
                "A little confident",
                "Somewhat confident",
                "Confident",
                "Very confident"
            ]
        },
        {
            "question_number": 5,
            "content": "How confident are you in finding information about your health problems from other sources?",
            "content_type": "assessment_question",
            "valid_responses": [
                "Not at all confident",
                "A little confident",
                "Somewhat confident",
                "Confident",
                "Very confident"
            ]
        },
    ],
}


# --- Core Functions ---
def initialize_document_store() -> WeaviateDocumentStore:
    """
    Initializes the WeaviateDocumentStore with the specified path and settings.

    Returns:
        An instance of WeaviateDocumentStore.
    """
    if weaviate_url := environ.get("WEAVIATE_URL"):
        if environ.get("WEAVIATE_API_KEY"):
            doc_store = WeaviateDocumentStore(url=weaviate_url, auth_client_secret=AuthApiKey())
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
    indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=embedding_model_name,
            progress_bar=True,
        ))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store))
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    logger.info("Created document embedding and writing pipeline.")
    return indexing_pipeline

def ingest_content(indexing_pipeline: Pipeline, content_flows: dict[str, list[dict[str, any]]]):
    """
    Processes content data, converts to Haystack Documents, and ingests them
    into the document store using the provided indexing pipeline.

    Args:
        indexing_pipeline: The Haystack pipeline for embedding and writing.
        content_flows: A dictionary where keys are flow_ids and values are lists
                       of content dictionaries (like ONBOARDING_FLOWS or ASSESSMENT_FLOWS).
    """
    documents_to_ingest: list[Document] = []
    flow_count = 0
    doc_count = 0

    for flow_id, content_pieces in content_flows.items():
        flow_count += 1
        for piece in content_pieces:
            doc_count += 1
            doc = Document(
                content=piece["content"],
                meta={
                    "flow_id": flow_id,
                    "question_number": piece["question_number"],
                    "content_type": piece["content_type"],
                }
            )
            documents_to_ingest.append(doc)
            logger.debug(f"Prepared document: flow='{flow_id}', seq={piece['question_number']}")

    if documents_to_ingest:
        logger.info(f"Starting ingestion of {doc_count} documents from {flow_count} flows...")
        indexing_pipeline.run({"embedder": {"documents": documents_to_ingest}})
        logger.info(f"Successfully ingested {doc_count} documents.")
    else:
        logger.warning("No documents found to ingest.")

def get_remaining_onboarding_questions(user_context: dict[str, any], all_onboarding_questions: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Identifies which onboarding questions still need to be asked based on user_context.
    """
    onboarding_data_to_collect = [
        "province", "area_type", "relationship_status", "education_level",
        "hunger_days", "num_children", "phone_ownership"
    ]
    collected_data_points = [k for k, v in user_context.items() if k in onboarding_data_to_collect and v is not None]
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
        ingest_content(indexing_pipe, ONBOARDING_FLOWS)
        logger.info("--- Ingesting Assessment Content ---")
        ingest_content(indexing_pipe, ASSESSMENT_FLOWS)
    else:
        logger.info("Documents found in the store. Skipping ingestion to avoid duplicates.")

    final_doc_count = document_store.count_documents()
    logger.info(f"Total documents in store after setup: {final_doc_count}")

    return document_store
