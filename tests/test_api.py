from fastapi.testclient import TestClient
from ai4gd_momconnect_haystack.api import app

def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"health": "ok"}
