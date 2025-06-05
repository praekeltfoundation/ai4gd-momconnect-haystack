import os
from unittest import mock
from ai4gd_momconnect_haystack.doc_store import initialize_document_store


@mock.patch.dict(os.environ, {"WEAVIATE_URL": "https://localhost:8080"}, clear=True)
def test_local_weaviate_noauth():
    doc_store = initialize_document_store()
    config = doc_store.to_dict()
    assert config["init_parameters"]["url"] == "https://localhost:8080"
    assert config["init_parameters"]["auth_client_secret"] is None
    assert config["init_parameters"]["embedded_options"] is None


@mock.patch.dict(
    os.environ,
    {"WEAVIATE_URL": "https://localhost:8080", "WEAVIATE_API_KEY": "testkey"},
    clear=True,
)
def test_local_weaviate_auth():
    doc_store = initialize_document_store()
    config = doc_store.to_dict()
    assert config["init_parameters"]["url"] == "https://localhost:8080"
    assert config["init_parameters"]["auth_client_secret"] is not None
    assert config["init_parameters"]["embedded_options"] is None


@mock.patch.dict(os.environ, {}, clear=True)
def test_embedded_weaviate():
    doc_store = initialize_document_store()
    config = doc_store.to_dict()
    assert config["init_parameters"]["url"] is None
    assert config["init_parameters"]["embedded_options"] is not None
