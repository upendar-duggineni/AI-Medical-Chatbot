// Small API client. In development Vite proxies /api to the FastAPI backend.
const BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    const err = new Error(
      "Can't reach the server. Make sure the backend is running (uvicorn main:app --reload)."
    );
    err.offline = true;
    throw err;
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let detail = "Something went wrong. Please try again.";
    if (typeof data.detail === "string") detail = data.detail;
    else if (res.status === 422) detail = "Please type between 2 and 500 characters.";
    throw new Error(detail);
  }
  return data;
}

export const sendMessage = (message, sessionId) =>
  request("/api/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

export const getStatus = () => request("/api/ask/status");
