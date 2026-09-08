# JobShield AI

JobShield AI is an end-to-end machine-learning system that identifies potentially fraudulent job postings. It uses TF-IDF text features, logistic regression, cost-sensitive evaluation, a versioned model artifact, and a FastAPI prediction service.

## Model Performance

Final results on the untouched test set:

| Metric | Result |
|---|---:|
| Precision | 70.78% |
| Recall | 85.16% |
| F1-score | 77.30% |
| ROC-AUC | 97.78% |
| PR-AUC | 84.79% |
| False positives | 45 |
| False negatives | 19 |

The decision threshold is `0.60`. Missing one fraudulent posting is provisionally treated as five times more costly than raising one false alert.

## ML Pipeline

```text
Job-posting fields
        ↓
Consistent text combination
        ↓
TF-IDF vectorization
        ↓
Logistic regression
        ↓
Fraud score
        ↓
Threshold-based classification