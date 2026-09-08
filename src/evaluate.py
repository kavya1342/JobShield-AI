import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "jobshield_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"


def main() -> None:
    test_df = pd.read_csv(TEST_PATH)

    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]
    model_version = model_bundle["version"]

    test_scores = model.predict_proba(x_test)[:, 1]

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions
    ).ravel()

    metrics = {
        "model_version": model_version,
        "threshold": threshold,
        "precision": float(
            precision_score(y_test, test_predictions)
        ),
        "recall": float(
            recall_score(y_test, test_predictions)
        ),
        "f1": float(
            f1_score(y_test, test_predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_scores)
        ),
        "pr_auc": float(
            average_precision_score(y_test, test_scores)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "error_cost": int(fp + 5 * fn),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("Final test evaluation completed")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()