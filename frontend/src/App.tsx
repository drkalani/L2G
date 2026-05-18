import { Link, Navigate, Route, Routes } from "react-router-dom";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { HomePage, JobsPage, ProjectsPage, ProjectWorkspace } from "./pages";
import { apiBaseDidChange, getApiBase, isColabCandidate, pingBackend, setApiBase } from "./api";

type BackendStatus = "unknown" | "checking" | "connected" | "failed";

function BackendConnectionBar() {
  const [backendUrl, setBackendUrl] = useState(getApiBase());
  const [status, setStatus] = useState<BackendStatus>("unknown");
  const [statusText, setStatusText] = useState("Not checked");
  const skipAutoRecheckRef = useRef(false);

  const statusColor =
    status === "connected"
      ? "var(--success)"
      : status === "failed"
      ? "var(--danger)"
      : status === "checking"
      ? "var(--warning)"
      : "var(--muted)";

  const testBackend = async (candidate: string | undefined) => {
    setStatus("checking");
    setStatusText("Checking...");
    try {
      await pingBackend(candidate);
      if (candidate !== undefined) {
        skipAutoRecheckRef.current = true;
        const normalized = setApiBase(candidate);
        setBackendUrl(normalized);
        skipAutoRecheckRef.current = false;
      }
      setStatus("connected");
      setStatusText("Connected");
    } catch (err) {
      setStatus("failed");
      setStatusText((err as Error).message || "Unable to connect");
    }
  };

  useEffect(() => {
    const sync = () => {
      setBackendUrl(getApiBase());
      if (!skipAutoRecheckRef.current) {
        void testBackend(undefined);
      } else {
        skipAutoRecheckRef.current = false;
      }
    };
    sync();
    const stop = apiBaseDidChange(sync);
    return stop;
  }, []);

  const hint = isColabCandidate(backendUrl)
    ? "Detected ngrok style URL. Make sure the full backend path is used (usually ends with /api)."
    : "Use /api for local Docker/hosted setups, or your Colab ngrok URL for remote execution.";

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Backend selector</h3>
      <div className="toolbar" style={{ alignItems: "flex-start" }}>
        <input
          value={backendUrl}
          onChange={(e) => setBackendUrl(e.target.value)}
          placeholder="/api or https://....ngrok-free.app/api"
          style={{ maxWidth: "560px", flex: 1 }}
        />
        <button
          className="btn"
          type="button"
          onClick={() => testBackend(backendUrl)}
          disabled={status === "checking"}
        >
          Save & test
        </button>
        <button className="btn" type="button" onClick={() => testBackend(undefined)}>
          Test current
        </button>
      </div>
      <p style={{ margin: "0.7rem 0 0.35rem", color: statusColor }}>
        Backend: <strong>{backendUrl}</strong> · {statusText}
      </p>
      <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.9rem" }}>{hint}</p>
    </div>
  );
}

function Layout({ children }: { children: ReactNode }) {
  return (
    <div>
      <header
        style={{
          borderBottom: "1px solid var(--border)",
          background: "rgba(26,35,50,0.85)",
          backdropFilter: "blur(8px)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        <div
          className="container"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            paddingTop: "0.85rem",
            paddingBottom: "0.85rem",
          }}
        >
          <Link to="/" style={{ fontWeight: 700, fontSize: "1.15rem" }}>
            L2G
          </Link>
          <nav style={{ display: "flex", gap: "1.25rem" }}>
            <Link to="/">Overview</Link>
            <Link to="/projects">Projects</Link>
            <Link to="/jobs">Job center</Link>
          </nav>
        </div>
      </header>
      <div className="container">
        <BackendConnectionBar />
      </div>
      <div className="container">{children}</div>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectWorkspace />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
