from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_version"] == "0.2.0"
    assert body["threshold"] == 0.60


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "title": "Work From Home Data Entry",
            "company_profile": "",
            "description": (
                "Earn money immediately. Pay a registration "
                "fee to receive your appointment letter."
            ),
            "requirements": "No experience required",
            "benefits": "Guaranteed weekly income",
        }
    )

    assert response.status_code == 200

    body = response.json()

    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["predicted_class"] in [0, 1]
    assert body["model_version"] == "0.2.0"


def test_missing_required_description_is_rejected():
    response = client.post(
        "/predict",
        json={
            "title": "Software Engineer"
        }
    )

    assert response.status_code == 422