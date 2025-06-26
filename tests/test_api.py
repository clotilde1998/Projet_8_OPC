# test_api.py
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app

from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_prediction_valid_client():
    response = client.get("/prediction/128180")  # Remplace par un ID existant
    assert response.status_code == 200
    assert "probability_default" in response.json()

def test_prediction_invalid_client():
    response = client.get("/prediction/999999999")  # ID inexistant
    assert response.status_code == 200
    assert "error" in response.json()

def test_shap_valid_client():
    response = client.get("/interpretabilite/128180")  # Remplace par un ID existant
    assert response.status_code == 200
    assert "shap_values" in response.json()

def test_shap_invalid_client():
    response = client.get("/interpretabilite/999999999")
    assert response.status_code == 200
    assert "error" in response.json()
