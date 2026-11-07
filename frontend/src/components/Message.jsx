
import ReactMarkdown from "react-markdown";

const RISK_LABEL = {
  emergency: "Urgent: call 112",
  see_doctor_soon: "See a doctor today",
  see_doctor: "See a doctor soon",
  self_care: "General information",
};

// The agent appends suggested next steps to its answer.
function parseAnswer(text = "") {
  const match = text.match(
    /^([\s\S]*?)\n\nSuggested next step: ([\s\S]*?)(?:\nWhy: ([\s\S]*))?$/
  );

  return match
    ? { body: match[1], next: match[2], why: match[3] }
    : { body: text };
}

export default function Message({ m, showTrace }) {
  const isBot = m.role === "bot";
  const { body, next, why } = isBot
    ? parseAnswer(m.text)
    : { body: m.text };

  const showDisclaimer =
    isBot &&
    m.disclaimer &&
    !["emergency", "followup"].includes(m.mode);

  return (
    <div className={`msg ${m.role} ${m.mode || ""}`}>
      {isBot && m.risk && m.mode !== "followup" && (
        <div className={`badge risk-${m.risk}`}>
          {RISK_LABEL[m.risk] || "General information"}
        </div>
      )}

      <div className="text">
        {isBot ? (
          <ReactMarkdown
            components={{
              a: ({ children, ...props }) => (
                <a {...props} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {body || ""}
          </ReactMarkdown>
        ) : (
          body
        )}
      </div>

      {next && (
        <div className="next">
          <strong>Suggested next step</strong>
          <span>{next}</span>
          {why && <small>Why: {why}</small>}
        </div>
      )}

      {m.sources?.length > 0 && (
        <div className="sources">
          <strong>Sources</strong>
          {m.sources.map((source, index) => (
            <a
              key={source.n ?? index}
              href={source.url}
              target="_blank"
              rel="noreferrer"
            >
              [{source.n ?? index + 1}] {source.title}
            </a>
          ))}
        </div>
      )}

      {showTrace && m.trace?.length > 0 && (
        <ol className="trace">
          {m.trace.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>
      )}

      {showDisclaimer && (
        <div className="note">{m.disclaimer}</div>
      )}
    </div>
  );
}
