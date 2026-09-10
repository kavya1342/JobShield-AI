const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

export async function analyseJobPosting(posting) {
  const response = await fetch(
    `${API_BASE_URL}/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(posting),
    }
  );

  if (!response.ok) {
    const errorBody = await response
      .json()
      .catch(() => null);

    throw new Error(
      errorBody?.detail?.[0]?.msg ||
        "Analysis failed"
    );
  }

  return response.json();
}