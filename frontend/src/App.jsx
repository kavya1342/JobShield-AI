import { useState } from "react";
import { analyseJobPosting } from "./api";
import "./App.css";

const emptyForm = {
  title: "",
  company_profile: "",
  description: "",
  requirements: "",
  benefits: "",
};

const examplePosting = {
  title: "Work From Home Data Entry",
  company_profile: "",
  description:
    "Earn money immediately. Pay a registration fee to receive your appointment letter.",
  requirements: "No experience required",
  benefits: "Guaranteed weekly income",
};

function App() {
  const [form, setForm] = useState(emptyForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setIsLoading(true);

    try {
      const prediction = await analyseJobPosting(form);
      setResult(prediction);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  function loadExample() {
    setForm(examplePosting);
    setResult(null);
    setError("");
  }

  function resetAnalysis() {
    setForm(emptyForm);
    setResult(null);
    setError("");
  }

  const maximumContribution = result
    ? Math.max(
        ...result.top_risk_terms.map(
          (item) => item.contribution
        ),
        0
      )
    : 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">J</div>
          <div>
            <p className="brand-name">JobShield AI</p>
            <p className="brand-detail">
              Job-posting risk analysis
            </p>
          </div>
        </div>

        <div className="model-status">
          <span className="status-dot" />
          Model v0.2.0
        </div>
      </header>

      <section className="workspace-heading">
        <p className="eyebrow">Fraud screening workspace</p>
        <h1>Analyse a job posting</h1>
        <p>
          Enter the posting exactly as it appeared.
          JobShield will identify suspicious patterns for
          human review.
        </p>
      </section>

      <div className="workspace-grid">
        <section className="panel form-panel">
          <div className="panel-heading">
            <div>
              <p className="step-label">Posting details</p>
              <h2>Job information</h2>
            </div>

            <button
              className="text-button"
              type="button"
              onClick={loadExample}
            >
              Load example
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <label>
              Job title
              <input
                name="title"
                value={form.title}
                onChange={handleChange}
                placeholder="e.g. Data Analyst"
                maxLength={500}
                required
              />
            </label>

            <label>
              Company profile
              <textarea
                name="company_profile"
                value={form.company_profile}
                onChange={handleChange}
                placeholder="Brief information about the employer"
                rows={3}
                maxLength={20000}
              />
            </label>

            <label>
              Job description
              <textarea
                name="description"
                value={form.description}
                onChange={handleChange}
                placeholder="Paste the complete job description"
                rows={6}
                maxLength={50000}
                required
              />
            </label>

            <div className="field-grid">
              <label>
                Requirements
                <textarea
                  name="requirements"
                  value={form.requirements}
                  onChange={handleChange}
                  placeholder="Skills and qualifications"
                  rows={4}
                  maxLength={30000}
                />
              </label>

              <label>
                Benefits
                <textarea
                  name="benefits"
                  value={form.benefits}
                  onChange={handleChange}
                  placeholder="Salary, benefits or incentives"
                  rows={4}
                  maxLength={30000}
                />
              </label>
            </div>

            <div className="form-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={resetAnalysis}
              >
                Clear
              </button>

              <button
                className="primary-button"
                type="submit"
                disabled={isLoading}
              >
                {isLoading
                  ? "Analysing…"
                  : "Analyse posting"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel result-panel">
          <div className="panel-heading">
            <div>
              <p className="step-label">Model assessment</p>
              <h2>Risk analysis</h2>
            </div>
          </div>

          {!result && !error && (
            <div className="empty-state">
              <div className="empty-icon">01</div>
              <h3>No analysis yet</h3>
              <p>
                Complete the required fields and analyse
                the posting to view its result.
              </p>
            </div>
          )}

          {error && (
            <div className="error-state" role="alert">
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div
                className={`decision-card ${
                  result.predicted_class === 1
                    ? "decision-risk"
                    : "decision-safe"
                }`}
              >
                <p>Assessment</p>
                <h3>
                  {result.predicted_class === 1
                    ? "Potential fraud risk"
                    : "Likely legitimate"}
                </h3>
                <span>
                  Review score:{" "}
                  {(result.fraud_score * 100).toFixed(1)}
                  /100
                </span>
              </div>

              <div className="score-row">
                <span>Decision threshold</span>
                <strong>
                  {(result.threshold * 100).toFixed(0)}
                  /100
                </strong>
              </div>

              <div className="signals">
                <div className="signals-heading">
                  <h3>Top model signals</h3>
                  <span>Contribution</span>
                </div>

                {result.top_risk_terms.map((item) => (
                  <div
                    className="signal-item"
                    key={item.term}
                  >
                    <div className="signal-copy">
                      <span>{item.term}</span>
                      <strong>
                        +{item.contribution.toFixed(3)}
                      </strong>
                    </div>
                    <div className="signal-track">
                      <div
                        className="signal-fill"
                        style={{
                          width: `${
                            maximumContribution > 0
                              ? (item.contribution /
                                  maximumContribution) *
                                100
                              : 0
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <p className="result-note">
                Signals show statistical associations learned
                from the training data. They are not proof of
                fraud and should support human review.
              </p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default App;