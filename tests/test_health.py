# tests/test_health.py
# check to see if fast server is responding or not
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_endpoint_exists():
    response = client.get("/docs")
    assert response.status_code == 200
