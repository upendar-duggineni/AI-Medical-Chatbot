import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import { getStatus, sendMessage } from "./api.js";

const WELCOME = {
  role: "bot",
  mode: "grounded",
  text:
    "Hi! Ask a general health question or describe your symptoms. I use MedlinePlus sources, " +
    "and I may ask a couple of follow-up questions. I can't diagnose or give medicine doses.",
  sources: [],
};

const EXAMPLES = [
  "I have a headache",
  "What are the symptoms of asthma?",
  "How is diabetes treated?",
  "My baby has a fever",
];

export default function App() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [status, setStatus] = useState(null); // null = checking
  const sessionRef = useRef(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  async function refreshStatus() {
    setStatus(null);
    try {
      const s = await getStatus();
      setStatus(s.ready ? { online: true, topics: s.topics } : { online: false, note: "Knowledge base not found" });
    } catch {
      setStatus({ online: false });
    }
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function newChat() {
    sessionRef.current = null;
    setMessages([WELCOME]);
    setInput("");
    inputRef.current?.focus();
  }

  async function send(text) {
    const question = (text ?? input).trim();
    if (question.length < 2 || loading) return;

    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendMessage(question, sessionRef.current);
      sessionRef.current = data.session_id;
      const flags = (data.red_flags || []).filter(Boolean);
      setMessages((m) => [
        ...m,
        // Fixed warnings from the backend's safety rules appear first, above the answer.
        ...(flags.length
          ? [{ role: "bot", mode: "warning", text: "Warning signs to know about\n\n" + flags.join("\n\n"), sources: [] }]
          : []),
        {
          role: "bot",
          mode: data.mode,
          text: data.answer,
          sources: data.sources || [],
          risk: data.risk,
          trace: data.trace || [],
          disclaimer: data.disclaimer,
        },
      ]);
      if (status && !status.online) refreshStatus();
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", mode: "error", text: err.message, sources: [] }]);
      if (err.offline) setStatus({ online: false });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  const pill =
    status === null
      ? { cls: "pill", label: "Checking…" }
      : status.online
      ? { cls: "pill ok", label: `Online · ${status.topics} topics` }
      : { cls: "pill bad", label: status.note || "Backend offline" };

  return (
    <div className="app">
      <div className="banner" role="note">
        Emergency? Call <strong>112</strong>. Feeling unsafe or in distress? Call Tele-MANAS{" "}
        <strong>14416</strong>.
      </div>

      <header className="topbar">
        <div>
          <h1>AI Health Assistant</h1>
          <p>General health information from MedlinePlus. Not medical advice.</p>
        </div>
        <div className="actions">
          <span className={pill.cls}>{pill.label}</span>
          {status && !status.online && (
            <button className="ghost" onClick={refreshStatus}>Retry</button>
          )}
          <label className="check">
            <input type="checkbox" checked={showTrace} onChange={(e) => setShowTrace(e.target.checked)} />
            Show agent steps
          </label>
          <button className="ghost" onClick={newChat}>New chat</button>
        </div>
      </header>

      <main className="chat" role="log" aria-live="polite" aria-label="Conversation">
        {messages.map((m, i) => (
          <Message key={i} m={m} showTrace={showTrace} />
        ))}
        {loading && <div className="msg bot typing">Thinking…</div>}

        {messages.length === 1 && (
          <div className="chips" aria-label="Example questions">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip" onClick={() => send(ex)}>
                {ex}
              </button>
            ))}
          </div>
        )}
        <div ref={endRef} />
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          ref={inputRef}
          value={input}
          maxLength={500}
          autoFocus
          aria-label="Your message"
          placeholder="Describe your symptoms or ask a health question"
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" disabled={loading || input.trim().length < 2}>
          Send
        </button>
      </form>

      <small className="attrib">
        Information from MedlinePlus.gov, U.S. National Library of Medicine.
      </small>
    </div>
  );
}
