import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from .config import (
    DECISION_THRESHOLD,
    MODEL_VERSION,
    TARGET_COLUMN,
)
from .model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    validation_path = PROCESSED_DATA_DIR / "validation.csv"

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    x_validation = validation_df.drop(
        columns=[TARGET_COLUMN]
    )
    y_validation = validation_df[TARGET_COLUMN]

    model = build_model()
    model.fit(x_train, y_train)

    validation_scores = model.predict_proba(
        x_validation
    )[:, 1]

    validation_predictions = (
        validation_scores >= DECISION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        validation_predictions
    ).ravel()

    metrics = {
        "threshold": DECISION_THRESHOLD,
        "precision": float(
            precision_score(
                y_validation,
                validation_predictions
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                validation_predictions
            )
        ),
        "f1": float(
            f1_score(
                y_validation,
                validation_predictions
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "threshold": DECISION_THRESHOLD,
        "version": MODEL_VERSION,
    }

    model_path = MODELS_DIR / "jobshield_model.joblib"
    metrics_path = REPORTS_DIR / "validation_metrics.json"

    joblib.dump(model_bundle, model_path)

    with metrics_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("Training completed")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()