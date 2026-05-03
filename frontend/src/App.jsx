import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";


/* ── Styles injected once ───────────────────────────────── */
const globalStyles = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e28;
    --border:    #252a38;
    --accent:    #5effa0;
    --accent2:   #4dc8ff;
    --accent3:   #ff6b6b;
    --text:      #e8eaf0;
    --text-dim:  #7a8099;
    --text-code: #b4f0c8;
    --radius:    12px;
    --radius-sm: 6px;
    --glow:      0 0 30px rgba(94, 255, 160, 0.12);
  }

  html, body, #root {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    line-height: 1.7;
  }

  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  @keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ── Markdown body ── */
  .md-body h1, .md-body h2, .md-body h3 {
    font-family: 'Syne', sans-serif; font-weight: 700; margin: 16px 0 8px; color: var(--text);
  }
  .md-body h1 { font-size: 1.2rem; }
  .md-body h2 { font-size: 1.05rem; }
  .md-body h3 { font-size: 0.95rem; color: var(--accent); }
  .md-body p  { margin-bottom: 10px; }
  .md-body ul, .md-body ol { padding-left: 20px; margin-bottom: 10px; }
  .md-body li { margin-bottom: 4px; }
  .md-body strong { color: var(--text); font-weight: 600; }
  .md-body em { color: var(--accent2); font-style: italic; }
  .md-body code {
    font-family: 'DM Mono', monospace; font-size: 0.82em;
    background: rgba(255,255,255,0.06); padding: 2px 6px;
    border-radius: 4px; color: var(--text-code);
  }
  .md-body pre {
    background: #0a0c12; border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 14px;
    overflow-x: auto; margin: 12px 0;
  }
  .md-body pre code { background: none; padding: 0; font-size: 0.8rem; }
  .md-body img {
    max-width: 100%; border-radius: var(--radius-sm);
    border: 1px solid var(--border); margin: 10px 0; display: block;
  }
  .md-body hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

  /* ── Table styles (used by both md-body and TableRenderer) ── */
  .md-table-wrap {
    width: 100%;
    overflow-x: auto;
    margin: 14px 0;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
  }
  .md-table-wrap table, .md-body table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .md-body table {
    /* override — table is now wrapped so we don't need border on the element */
    border: none;
    margin: 0;
  }
  .md-table-wrap thead, .md-body thead {
    background: linear-gradient(135deg, rgba(94,255,160,0.1), rgba(77,200,255,0.1));
  }
  .md-table-wrap th, .md-body th {
    padding: 10px 14px;
    text-align: left;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    font-weight: 400;
    white-space: nowrap;
  }
  .md-table-wrap td, .md-body td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(37,42,56,0.6);
    color: var(--text-dim);
    vertical-align: top;
    line-height: 1.5;
  }
  /* Allow <br> inside cells to render properly */
  .md-table-wrap td br, .md-body td br { display: block; content: ""; margin: 2px 0; }
  .md-table-wrap tr:last-child td, .md-body tr:last-child td { border-bottom: none; }
  .md-table-wrap tr:hover td, .md-body tr:hover td { background: rgba(255,255,255,0.02); }

  /* Section label above a table block */
  .table-section-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--accent2);
    margin: 18px 0 6px;
    padding-left: 2px;
  }
`;

/* ── Markdown renderer using marked (if available) ────────
   Falls back to a smarter regex path that correctly handles
   multi-line table cells containing <br/>.                  */
function renderMarkdown(text) {
  if (!text) return "";

  // Use marked if loaded via CDN script tag
  if (typeof window !== "undefined" && window.marked) {
    try {
      window.marked.setOptions({ breaks: true, gfm: true });
      const html = window.marked.parse(text);
      // Wrap every <table> in a scroll container
      return html.replace(/<table>/g, '<div class="md-table-wrap"><table>')
                 .replace(/<\/table>/g, "</table></div>");
    } catch {
      // fall through to manual renderer
    }
  }

  // ── Manual renderer (fallback) ──────────────────────────
  // Handles tables first, before generic line transforms
  const lines = text.split("\n");
  const output = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Detect a markdown table: | col | col |
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[-:| ]+\|/.test(lines[i + 1])) {
      // Collect all table lines
      const tableLines = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        tableLines.push(lines[i]);
        i++;
      }

      const parseRow = (row) =>
        row
          .replace(/^\s*\|/, "")
          .replace(/\|\s*$/, "")
          .split("|")
          .map(cell => cell.trim());

      const headers = parseRow(tableLines[0]);
      // tableLines[1] is the separator row — skip it
      const rows = tableLines.slice(2).map(parseRow);

      const thHTML = headers
        .map(h => `<th>${h}</th>`)
        .join("");
      const tbodyHTML = rows
        .map(cells => {
          const tdHTML = cells
            .map(c => `<td>${c.replace(/\\n|<br\s*\/?>/gi, "<br/>")}</td>`)
            .join("");
          return `<tr>${tdHTML}</tr>`;
        })
        .join("");

      output.push(
        `<div class="md-table-wrap"><table><thead><tr>${thHTML}</tr></thead><tbody>${tbodyHTML}</tbody></table></div>`
      );
      continue;
    }

    // Headings
    const h3 = line.match(/^### (.+)/);
    if (h3) { output.push(`<h3>${h3[1]}</h3>`); i++; continue; }
    const h2 = line.match(/^## (.+)/);
    if (h2) { output.push(`<h2>${h2[1]}</h2>`); i++; continue; }
    const h1 = line.match(/^# (.+)/);
    if (h1) { output.push(`<h1>${h1[1]}</h1>`); i++; continue; }

    // HR
    if (/^---$/.test(line)) { output.push("<hr/>"); i++; continue; }

    // List item
    if (/^\s*[-*] (.+)/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*] (.+)/.test(lines[i])) {
        items.push(`<li>${lines[i].replace(/^\s*[-*] /, "")}</li>`);
        i++;
      }
      output.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    // Blank line → paragraph break
    if (line.trim() === "") { output.push("<br/>"); i++; continue; }

    // Normal text line — apply inline formatting
    const formatted = line
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
    output.push(`<p>${formatted}</p>`);
    i++;
  }

  return output.join("\n");
}

/* ── TableRenderer: robustly renders any table format ────
   Priority: marked (if loaded) → pipe markdown → tab-separated */
function TableRenderer({ markdown }) {
  if (!markdown) return null;

  // ── 1. marked (CDN) ──────────────────────────────────────
  if (typeof window !== "undefined" && window.marked) {
    try {
      window.marked.setOptions({ breaks: true, gfm: true });
      const html = window.marked.parse(markdown);
      return (
        <div
          className="md-table-wrap"
          dangerouslySetInnerHTML={{ __html: html }}
          style={{ marginTop: 8 }}
        />
      );
    } catch { /* fall through */ }
  }

  // ── 2. Detect format and parse to { headers, rows } ─────
  const lines = markdown.trim().split("\n").filter(Boolean);

  let headers = [];
  let rows = [];

  const isPipeLine = (l) => l.includes("|");
  const isSepLine  = (l) => /^[\s|:\-]+$/.test(l);

  if (lines.length >= 2 && isPipeLine(lines[0])) {
    // ── Pipe-markdown format ──────────────────────────────
    const parseRow = (l) =>
      l.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(c => c.trim());

    headers = parseRow(lines[0]);
    // lines[1] is separator — skip it
    rows = lines
      .slice(2)
      .filter(l => !isSepLine(l) && isPipeLine(l))
      .map(parseRow);

  } else {
    // ── Tab / mixed-whitespace format (fallback) ──────────
    // Group lines into header + body by looking for a blank separator
    // or by treating the first line as header if no separator exists.
    const tabParse = (l) => l.split(/\t|  {2,}/).map(c => c.trim()).filter(Boolean);

    let bodyStart = 1;
    headers = tabParse(lines[0]);

    // Skip a separator-like line if present
    if (lines[1] && isSepLine(lines[1])) bodyStart = 2;

    rows = lines.slice(bodyStart).map(tabParse);
  }

  if (!headers.length) {
    return <pre style={{ fontSize: "0.8rem", color: "var(--text-dim)", whiteSpace: "pre-wrap" }}>{markdown}</pre>;
  }

  return (
    <div className="md-table-wrap" style={{ marginTop: 8 }}>
      <table>
        <thead>
          <tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((cells, ri) => (
            <tr key={ri}>
              {/* Pad short rows to header width */}
              {Array.from({ length: headers.length }, (_, ci) => (
                <td
                  key={ci}
                  dangerouslySetInnerHTML={{
                    // Render <br> and \n as line breaks inside cells
                    __html: (cells[ci] ?? "")
                      .replace(/\\n/g, "<br>")
                      .replace(/\n/g, "<br>"),
                  }}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── TablesSection: renders all tables returned in a response */
function TablesSection({ tables }) {
  if (!tables?.length) return null;
  return (
    <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
      <div style={{
        fontSize: "0.68rem", fontFamily: "'DM Mono', monospace",
        color: "var(--text-dim)", letterSpacing: 1,
        textTransform: "uppercase", marginBottom: 10,
      }}>
        Referenced Tables
      </div>
      {tables.map((t, i) => (
        <div key={i}>
          {tables.length > 1 && (
            <div className="table-section-label">Table {i + 1}</div>
          )}
          <TableRenderer markdown={t} />
        </div>
      ))}
    </div>
  );
}

/* ── Sub-components ─────────────────────────────────────── */

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: 5, alignItems: "center", padding: "4px 0" }}>
      {[0, 0.2, 0.4].map((delay, i) => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: "50%",
          background: "var(--accent)",
          display: "inline-block",
          animation: `bounce 1.2s ${delay}s infinite`,
        }} />
      ))}
    </div>
  );
}

function SourceCards({ sources }) {
  if (!sources?.length) return null;
  return (
    <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
      <div style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", color: "var(--text-dim)", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
        Sources
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {sources.slice(0, 4).map((s, i) => (
          <div key={i} style={{
            background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)", padding: "8px 12px",
            fontSize: "0.75rem", flex: "1", minWidth: 180,
          }}>
            <div style={{ fontFamily: "'DM Mono', monospace", color: "var(--accent2)", marginBottom: 3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              📄 {s.file_name}
            </div>
            <div style={{ color: "var(--text-dim)", lineHeight: 1.4 }}>
              Page {s.page} — {s.snippet?.slice(0, 100)}…
            </div>
            <div style={{ marginTop: 4, fontFamily: "'DM Mono', monospace", color: "var(--text-dim)", fontSize: "0.68rem" }}>
              relevance: {s.score}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImageGrid({ images }) {
  if (!images?.length) return null;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10, marginTop: 12 }}>
      {images.map((img, i) => (
        <div key={i} style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
          <img src={`${API}/images/${img}`} alt={img} style={{ width: "100%", display: "block", objectFit: "contain", maxHeight: 140 }} />
          <div style={{ fontSize: 10, color: "var(--text-dim)", padding: "4px 8px", borderTop: "1px solid var(--border)", wordBreak: "break-all", fontFamily: "'DM Mono', monospace" }}>{img}</div>
        </div>
      ))}
    </div>
  );
}

function Message({ role, content, sources, images, tables, hasTables, hasImages, isTyping }) {
  const isUser = role === "user";
  const isError = role === "error";

  const bubbleStyle = {
    background: isUser
      ? "rgba(94,255,160,0.06)"
      : isError
      ? "rgba(255,107,107,0.08)"
      : "var(--surface2)",
    border: `1px solid ${isUser ? "rgba(94,255,160,0.2)" : isError ? "rgba(255,107,107,0.3)" : "var(--border)"}`,
    borderRadius: "var(--radius)",
    padding: "16px 18px",
    fontSize: "0.88rem",
    lineHeight: 1.75,
    color: isError ? "var(--accent3)" : "var(--text)",
  };

  return (
    <div style={{ display: "flex", gap: 14, animation: "fadeSlide 0.3s ease" }}>
      <div style={{
        width: 34, height: 34, borderRadius: 8,
        display: "grid", placeItems: "center",
        fontSize: "0.9rem", flexShrink: 0, marginTop: 2,
        background: isUser
          ? "linear-gradient(135deg,#667eea,#764ba2)"
          : "linear-gradient(135deg,var(--accent),var(--accent2))",
      }}>
        {isUser ? "👤" : "🤖"}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: "0.72rem", fontFamily: "'DM Mono', monospace", color: "var(--text-dim)", marginBottom: 6, letterSpacing: "0.5px" }}>
          {isUser ? "YOU" : "TECHDOC AI"}
        </div>
        <div style={bubbleStyle}>
          {isTyping ? (
            <TypingIndicator />
          ) : isUser ? (
            <span>{content}</span>
          ) : (
            <>
              {(hasTables || hasImages) && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
                  {hasTables && <span style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", padding: "2px 8px", borderRadius: 20, border: "1px solid var(--accent2)", color: "var(--accent2)" }}>⊞ Contains Table</span>}
                  {hasImages && <span style={{ fontSize: "0.68rem", fontFamily: "'DM Mono', monospace", padding: "2px 8px", borderRadius: 20, border: "1px solid var(--accent)", color: "var(--accent)" }}>🖼 Contains Image</span>}
                </div>
              )}
              {/* LLM answer text */}
              <div className="md-body" dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />
              {/* Actual table data returned from the backend — rendered properly */}
              <TablesSection tables={tables} />
              {/* Images */}
              <ImageGrid images={images} />
              {/* Source cards */}
              <SourceCards sources={sources} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Upload modal ───────────────────────────────────────── */
function UploadModal({ onClose, onUploaded }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");

  async function doUpload() {
    if (!file) return;
    setStatus("loading");
    const fd = new FormData();
    fd.append("file", file);
    try {
      await fetch(`${API}/upload`, { method: "POST", body: fd });
      setStatus("ok");
      onUploaded(file.name);
    } catch {
      setStatus("err");
    }
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }}>
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: 28, width: 420, maxWidth: "90vw" }}>
        <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: "1rem", marginBottom: 20 }}>Upload PDF</div>

        <label style={{
          display: "block", border: "1px dashed var(--border)", borderRadius: "var(--radius-sm)",
          padding: "2rem 1rem", textAlign: "center", cursor: "pointer",
          background: file ? "rgba(94,255,160,0.04)" : "transparent",
          transition: "all 0.2s",
        }}>
          <input type="file" accept="application/pdf" style={{ display: "none" }} onChange={e => setFile(e.target.files[0])} />
          <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.5 }}>📄</div>
          <div style={{ fontSize: 13, color: file ? "var(--accent)" : "var(--text-dim)" }}>
            {file ? file.name : "Click to choose a PDF"}
          </div>
        </label>

        <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "flex-end" }}>
          <button onClick={onClose} style={outlineBtn}>Cancel</button>
          <button onClick={doUpload} disabled={!file || status === "loading"} style={primaryBtn(!file || status === "loading")}>
            {status === "loading" ? "Uploading…" : status === "ok" ? "Done ✓" : "Upload"}
          </button>
        </div>
        {status === "err" && <div style={{ marginTop: 10, color: "var(--accent3)", fontSize: 12 }}>Upload failed. Check the server.</div>}
      </div>
    </div>
  );
}

const outlineBtn = {
  padding: "8px 16px", fontSize: 13, border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)", background: "transparent",
  color: "var(--text-dim)", cursor: "pointer",
};
const primaryBtn = (disabled) => ({
  padding: "8px 18px", fontSize: 13, border: "none",
  borderRadius: "var(--radius-sm)", cursor: disabled ? "not-allowed" : "pointer",
  background: disabled ? "rgba(94,255,160,0.2)" : "linear-gradient(135deg,var(--accent),var(--accent2))",
  color: disabled ? "var(--text-dim)" : "#0d0f14",
  fontWeight: 600,
  opacity: disabled ? 0.6 : 1,
});

/* ── Main App ────────────────────────────────────────────── */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState({ checking: true, ready: false, docs: 0, text: "Checking…" });
  const [showUpload, setShowUpload] = useState(false);
  const chatRef = useRef(null);
  const textareaRef = useRef(null);

  /* inject global styles once */
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = globalStyles;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);

  /* health check */
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/health`);
        const d = await r.json();
        setSystemStatus({ ready: true, docs: d.cache_files ?? 0, text: "RAG system ready" });
      } catch {
        setSystemStatus({ ready: false, docs: 0, text: "Server unreachable" });
      }
    })();
  }, []);

  /* scroll to bottom */
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  async function sendQuestion(q) {
    const text = (q || query).trim();
    if (!text || loading) return;
    setQuery("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setMessages(m => [...m, { role: "user", content: text }, { role: "typing" }]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      const data = await res.json();
      setMessages(m => [
        ...m.filter(x => x.role !== "typing"),
        {
          role: "ai",
          content: data.answer,
          sources: data.sources,
          images: data.images,
          // tables is now an array of raw markdown strings from the backend
          tables: data.tables ?? [],
          hasTables: data.has_tables,
          hasImages: data.has_images,
        },
      ]);
    } catch (e) {
      setMessages(m => [...m.filter(x => x.role !== "typing"), { role: "error", content: `Network error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={(name) => {
            setSystemStatus(s => ({ ...s, ready: true, text: `Indexed: ${name}` }));
            setTimeout(() => setShowUpload(false), 800);
          }}
        />
      )}

      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", height: "100vh", width: "100vw" }}>

        {/* ── Sidebar ── */}
        <aside style={{
          background: "var(--surface)", borderRight: "1px solid var(--border)",
          display: "flex", flexDirection: "column", padding: "28px 20px", gap: 24, overflowY: "auto",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 34, height: 34, background: "linear-gradient(135deg,var(--accent),var(--accent2))", borderRadius: 8, display: "grid", placeItems: "center", fontSize: "1rem", flexShrink: 0 }}>📘</div>
            <div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "1.1rem", letterSpacing: "-0.5px" }}>TechDoc AI</div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.5px" }}>manual assistant v1.0</div>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: "0.65rem", fontFamily: "'DM Mono', monospace", letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--text-dim)" }}>System Status</div>
            <div style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", padding: "12px 14px", display: "flex", alignItems: "center", gap: 10, fontSize: "0.82rem" }}>
              <span style={{
                width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
                background: systemStatus.ready ? "var(--accent)" : "#555",
                boxShadow: systemStatus.ready ? "0 0 8px var(--accent)" : "none",
                display: "inline-block",
                animation: systemStatus.checking ? "pulse 1s infinite" : "none",
              }} />
              {systemStatus.text}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: "0.65rem", fontFamily: "'DM Mono', monospace", letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--text-dim)" }}>Documents</div>
            <button onClick={() => setShowUpload(true)} style={{
              background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
              padding: "10px 14px", color: "var(--text-dim)", fontSize: "0.82rem",
              cursor: "pointer", textAlign: "left", transition: "all 0.2s", fontFamily: "'DM Sans', sans-serif",
            }}
              onMouseEnter={e => { e.target.style.borderColor = "var(--accent)"; e.target.style.color = "var(--text)"; }}
              onMouseLeave={e => { e.target.style.borderColor = "var(--border)"; e.target.style.color = "var(--text-dim)"; }}
            >
              + Upload PDF
            </button>
          </div>

          

          <div style={{ marginTop: "auto", fontSize: "0.72rem", color: "var(--text-dim)", fontFamily: "'DM Mono', monospace", borderTop: "1px solid var(--border)", paddingTop: 18 }}>
            RAG · LlamaIndex · FastAPI<br />HuggingFace Embeddings
          </div>
        </aside>

        {/* ── Main ── */}
        <div style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>

          <div style={{ borderBottom: "1px solid var(--border)", padding: "16px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0, background: "var(--surface)" }}>
            <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: "1rem", color: "var(--text-dim)" }}>
              Technical <span style={{ color: "var(--text)" }}>Manual Q&A</span>
            </div>
            <div style={{
              fontFamily: "'DM Mono', monospace", fontSize: "0.68rem", padding: "3px 10px",
              borderRadius: 20, letterSpacing: "0.5px",
              border: systemStatus.ready ? "1px solid var(--accent)" : "1px solid var(--border)",
              color: systemStatus.ready ? "var(--accent)" : "var(--text-dim)",
              background: systemStatus.ready ? "rgba(94,255,160,0.08)" : "transparent",
            }}>
              {systemStatus.ready ? `${systemStatus.docs} doc(s) indexed` : "offline"}
            </div>
          </div>

          <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: 28, display: "flex", flexDirection: "column", gap: 28, scrollBehavior: "smooth" }}>
            {messages.length === 0 ? (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: 16, padding: 40 }}>
                <div style={{ fontSize: "3.5rem", lineHeight: 1, filter: "drop-shadow(0 0 20px rgba(94,255,160,0.3))" }}>🔍</div>
                <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "1.8rem", background: "linear-gradient(135deg,var(--accent),var(--accent2))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                  Ask Your Manual
                </div>
                <div style={{ color: "var(--text-dim)", maxWidth: 400, fontSize: "0.9rem" }}>
                  Ask any question about the technical manuals. Tables and images from the documents will be displayed directly in the answer.
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 28, textAlign: "left" }}>
                {messages.map((msg, i) =>
                  msg.role === "typing" ? (
                    <Message key={i} role="ai" isTyping />
                  ) : (
                    <Message key={i} {...msg} />
                  )
                )}
              </div>
            )}
          </div>

          <div style={{ padding: "20px 28px 24px", borderTop: "1px solid var(--border)", background: "var(--surface)", flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: "10px 10px 10px 18px", transition: "border-color 0.2s, box-shadow 0.2s" }}
              onFocus={e => { e.currentTarget.style.borderColor = "rgba(94,255,160,0.4)"; e.currentTarget.style.boxShadow = "var(--glow)"; }}
              onBlur={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.boxShadow = "none"; }}
            >
              <textarea
                ref={textareaRef}
                value={query}
                onChange={e => { setQuery(e.target.value); autoResize(e.target); }}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); } }}
                placeholder="Ask a question about the technical manuals…"
                rows={1}
                style={{ flex: 1, background: "none", border: "none", outline: "none", color: "var(--text)", fontFamily: "'DM Sans', sans-serif", fontSize: "0.9rem", resize: "none", maxHeight: 120, minHeight: 24, lineHeight: 1.5 }}
              />
              <button
                onClick={() => sendQuestion()}
                disabled={loading || !query.trim()}
                style={{ width: 38, height: 38, borderRadius: 8, border: "none", background: "linear-gradient(135deg,var(--accent),var(--accent2))", color: "#0d0f14", cursor: loading || !query.trim() ? "not-allowed" : "pointer", display: "grid", placeItems: "center", fontSize: "1rem", flexShrink: 0, transition: "opacity 0.2s, transform 0.1s", opacity: loading || !query.trim() ? 0.35 : 1 }}
              >
                ➤
              </button>
            </div>
            <div style={{ marginTop: 8, fontSize: "0.7rem", color: "var(--text-dim)", fontFamily: "'DM Mono', monospace", textAlign: "center" }}>
              Enter to send · Shift+Enter for new line
            </div>
          </div>
        </div>
      </div>
    </>
  );
}