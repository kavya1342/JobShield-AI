# JobShield AI — Model Scope

## Intended purpose

JobShield AI estimates whether a job advertisement contains patterns associated with fraudulent postings. It is intended to provide a warning for human review, not to automatically reject or remove a job posting.

## Current dataset

The initial model is trained using the Employment Scam Aegean Dataset (EMSCAD), containing English-language job advertisements published between 2012 and 2014.

## Current supported task

- Input: Job title, company profile, description, requirements and benefits
- Output: Fraud-risk score and warning classification
- Task type: Supervised binary classification
- Positive class: Fraudulent posting

## Current limitations

- Training data is old and may not represent modern scams.
- The dataset contains relatively few fraudulent examples.
- Modern terminology and communication platforms may be missing.
- Performance on Indian job and internship postings is unknown.
- Model scores are not yet probability-calibrated.
- Predictions must not be treated as proof of fraud.

## Current use policy

The model may assist users by identifying suspicious patterns. A human should verify the company and posting before making a final decision.