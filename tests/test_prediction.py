
from src.model import build_model
from src.predict import predict_posting

def test_prediction_output_contract():
    posting = {
        "title": "Software Engineer",
        "company_profile": "Technology company",
        "description": "Develop and maintain Python applications",
        "requirements": "Python and SQL",
        "benefits": "Health insurance",
    }

    result = predict_posting(posting)

    assert 0.0 <= result["fraud_score"] <= 1.0
    assert result["predicted_class"] in [0, 1]
    assert result["prediction"] in [
        "likely_legitimate",
        "potentially_fraudulent",
    ]
    assert result["threshold"] == 0.60
    assert result["model_version"] == "0.2.0"


def test_missing_optional_fields_do_not_crash():
    posting = {
        "title": "Data Analyst",
        "description": "Analyse business data using SQL",
    }

    result = predict_posting(posting)

    assert 0.0 <= result["fraud_score"] <= 1.0
    assert result["predicted_class"] in [0, 1]


def test_empty_text_does_not_crash():
    posting = {
        "title": "",
        "description": "",
    }

    result = predict_posting(posting)

    assert 0.0 <= result["fraud_score"] <= 1.0

def test_negation_words_are_preserved():
    model = build_model()
    stop_words = model.named_steps["tfidf"].stop_words

    for word in ["no", "not", "nor", "never"]:
        assert word not in stop_words

def test_risk_terms_are_positive_and_ranked():
    posting = {
        "title": "Work From Home Data Entry",
        "description": (
            "Earn money immediately and pay a "
            "registration fee."
        ),
    }

    result = predict_posting(posting)
    risk_terms = result["top_risk_terms"]

    assert 1 <= len(risk_terms) <= 5

    contributions = [
        item["contribution"]
        for item in risk_terms
    ]

    assert all(
        contribution > 0
        for contribution in contributions
    )

    assert contributions == sorted(
        contributions,
        reverse=True
    )