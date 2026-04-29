import { useEffect, useMemo, useState } from "react";
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useLocalStorage } from "./hooks/useLocalStorage";
import { useNetworkStatus } from "./hooks/useNetworkStatus";

const FEATURE_GROUPS = [
  {
    title: "CAIDA attributes for ISP A",
    fields: [
      { key: "caida_rank_1", label: "CAIDA Rank" },
      { key: "caida_longitude_1", label: "Longitude" },
      { key: "caida_latitude_1", label: "Latitude" },
      { key: "caida_customer_1", label: "Customer Count" },
      { key: "caida_peer_1", label: "Peer Count" },
      { key: "caida_provider_1", label: "Provider Count" },
      { key: "caida_total_1", label: "Total Count" },
      { key: "caida_NumberASNs_1", label: "Number of ASNs" },
      { key: "caida_NumberPrefix_1", label: "Number of Prefixes" },
      { key: "caida_NumberAddrs_1", label: "Number of Addresses" },
      { key: "caida_Country_1", label: "Country Code" }
    ]
  },
  {
    title: "PeeringDB attributes for ISP A",
    fields: [
      { key: "pdb_fac_count_1", label: "Facility Count" },
      { key: "pdb_id_1", label: "PeeringDB ID" },
      { key: "pdb_info_traffic_1", label: "Traffic Level" },
      { key: "pdb_ix_count_1", label: "IX Count" },
      { key: "pdb_info_prefixes6_1", label: "IPv6 Prefix Count" },
      { key: "pdb_info_prefixes4_1", label: "IPv4 Prefix Count" }
    ]
  },
  {
    title: "CAIDA attributes for ISP B",
    fields: [
      { key: "caida_rank_2", label: "CAIDA Rank" },
      { key: "caida_longitude_2", label: "Longitude" },
      { key: "caida_latitude_2", label: "Latitude" },
      { key: "caida_customer_2", label: "Customer Count" },
      { key: "caida_peer_2", label: "Peer Count" },
      { key: "caida_provider_2", label: "Provider Count" },
      { key: "caida_total_2", label: "Total Count" },
      { key: "caida_NumberASNs_2", label: "Number of ASNs" },
      { key: "caida_NumberPrefix_2", label: "Number of Prefixes" },
      { key: "caida_NumberAddrs_2", label: "Number of Addresses" },
      { key: "caida_Country_2", label: "Country Code" }
    ]
  },
  {
    title: "PeeringDB attributes for ISP B",
    fields: [
      { key: "pdb_fac_count_2", label: "Facility Count" },
      { key: "pdb_id_2", label: "PeeringDB ID" },
      { key: "pdb_info_traffic_2", label: "Traffic Level" },
      { key: "pdb_ix_count_2", label: "IX Count" },
      { key: "pdb_info_prefixes6_2", label: "IPv6 Prefix Count" },
      { key: "pdb_info_prefixes4_2", label: "IPv4 Prefix Count" }
    ]
  },
  {
    title: "Derived pair features",
    fields: [
      { key: "ConeOverlap", label: "Cone Overlap" },
      { key: "AffinityScore", label: "Affinity Score" }
    ]
  }
];

const FEATURE_FIELDS = FEATURE_GROUPS.flatMap((group) => group.fields);
const DATASET_META_FIELDS = ["asn1", "asn2"];
const DATASET_COLUMNS = [...DATASET_META_FIELDS, ...FEATURE_FIELDS.map((field) => field.key)];

const DEFAULT_MODEL_VALUE = "xgboost";
const DEFAULT_MODEL_LABEL = "XGBoost (optimal_pus_features.pkl)";

// ===== LOCAL BACKEND (FastAPI) =====
// const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ===== HUGGING FACE SPACE =====
const API_BASE = import.meta.env.VITE_API_BASE_URL || "https://naveenyonko-sem6mini.hf.space";

const DEFAULT_DATASET_VALUES = DATASET_COLUMNS.reduce((acc, key) => {
  acc[key] = DATASET_META_FIELDS.includes(key) ? "" : 0;
  return acc;
}, {});

const COLUMN_ALIAS_MAP = buildColumnAliasMap();

function buildInitialDatasetRow() {
  return { ...DEFAULT_DATASET_VALUES };
}

function buildColumnAliasMap() {
  const map = new Map();

  function addAlias(alias, targetKey) {
    const normalized = normalizeHeader(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, targetKey);
    }
  }

  for (const key of DATASET_META_FIELDS) {
    addAlias(key, key);
  }

  for (const field of FEATURE_FIELDS) {
    const baseKey = field.key.replace(/_[12]$/, "");
    addAlias(field.key, field.key);
    addAlias(baseKey, field.key);
    addAlias(field.label, field.key);
  }

  return map;
}

function normalizeHeader(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function splitCsvLine(line) {
  const cells = [];
  let current = "";
  let insideQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];

    if (character === '"') {
      if (insideQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
      continue;
    }

    if (character === "," && !insideQuotes) {
      cells.push(current);
      current = "";
      continue;
    }

    current += character;
  }

  cells.push(current);
  return cells;
}

function parseCsvText(text) {
  const cleaned = text.replace(/^\uFEFF/, "").trim();

  if (!cleaned) {
    throw new Error("The CSV file is empty.");
  }

  const lines = cleaned.split(/\r?\n/).filter((line) => line.trim() !== "");

  if (lines.length < 2) {
    throw new Error("The CSV needs a header row and at least one data row.");
  }

  const headers = splitCsvLine(lines[0]).map((header) => header.trim());
  const dataRow = splitCsvLine(lines[1]);

  return headers.reduce((acc, header, index) => {
    acc[header] = dataRow[index] ?? "";
    return acc;
  }, {});
}

function parseFeatureValue(value, fallback = 0) {
  if (value === null || value === undefined) {
    return fallback;
  }

  const text = String(value).trim();
  if (text === "") {
    return fallback;
  }

  const numeric = Number(text);
  if (!Number.isFinite(numeric)) {
    throw new Error(`Invalid numeric value: ${text}`);
  }

  return numeric;
}

function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Unable to read the selected CSV file."));
    reader.readAsText(file);
  });
}

function applyDatasetRow(row) {
  const resolved = { ...DEFAULT_DATASET_VALUES };
  let matchedColumns = 0;

  for (const [header, rawValue] of Object.entries(row)) {
    const normalizedHeader = normalizeHeader(header);
    const targetKey = COLUMN_ALIAS_MAP.get(normalizedHeader);

    if (!targetKey) {
      continue;
    }

    if (DATASET_META_FIELDS.includes(targetKey)) {
      resolved[targetKey] = String(rawValue ?? "").trim();
    } else {
      resolved[targetKey] = parseFeatureValue(rawValue, 0);
    }

    matchedColumns += 1;
  }

  return { resolved, matchedColumns };
}

export default function App() {
  const isOnline = useNetworkStatus();
  const [history, setHistory] = useLocalStorage("prediction_history", []);
  const [theme, setTheme] = useLocalStorage("theme", "light");
  const [result, setResult] = useState(() => history[0] || null);
  const [apiStatus, setApiStatus] = useState({ state: "checking", detail: "Checking backend..." });

  useEffect(() => {
    let isMounted = true;
    let intervalId = null;

    async function checkApi() {
      try {
        // ===== LOCAL BACKEND =====
        // const response = await fetch(`${API_BASE}/health`);
        // const data = await response.json();
        //
        // if (response.ok) {
        //   setApiStatus({
        //     state: "ready",
        //     detail: `${data.modelsLoaded.length} models loaded, ${data.featureCount} features ready`
        //   });
        //   return;
        // }

        // ===== HUGGING FACE (Gradio) =====
        const response = await fetch(`${API_BASE}/config`);

        if (!isMounted) return;

        if (response.ok) {
          setApiStatus({ state: "ready", detail: "Gradio API reachable" });
          return;
        }

        setApiStatus({ state: "degraded", detail: `Gradio config responded with status ${response.status}` });
      } catch {
        if (isMounted) {
          setApiStatus({ state: "offline", detail: "API not reachable" });
        }
      }
    }

    checkApi();
    intervalId = window.setInterval(checkApi, 5000);

    return () => {
      isMounted = false;
      if (intervalId) {
        window.clearInterval(intervalId);
      }
    };
  }, []);

  useEffect(() => {
    try {
      document.documentElement.setAttribute("data-theme", theme);
    } catch {
      // ignore
    }
  }, [theme]);

  function handleSaveResult(nextResult) {
    setResult(nextResult);
    setHistory((prev) => [nextResult, ...prev].slice(0, 20));
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="hero">
          <p className="kicker">Peering Partner Prediction</p>
          <h1>Predict Link Potential Between Two ISPs</h1>
          <p className="subtext">
            React talks to a Hugging Face Space API and returns the peering label plus confidence.
          </p>
          <button
            className="theme-toggle"
            aria-label="Toggle theme"
            onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>
          <div className="status-stack">
            <span className={`status ${isOnline ? "online" : "offline"}`}>{isOnline ? "Browser online" : "Browser offline"}</span>
            <span className={`status ${apiStatus.state}`}>API: {apiStatus.detail}</span>
          </div>
          <nav className="top-nav" aria-label="Main navigation">
            <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Home
            </NavLink>
            <NavLink to="/results" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Results
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              About
            </NavLink>
          </nav>
        </header>

        <BreadcrumbBar />
        <AnimatedRoutes isOnline={isOnline} apiStatus={apiStatus} onSaveResult={handleSaveResult} result={result} history={history} />
      </div>
    </BrowserRouter>
  );
}

function BreadcrumbBar() {
  const location = useLocation();

  return (
    <nav className="breadcrumbs" aria-label="Breadcrumb">
      <Link className={`crumb ${location.pathname === "/" ? "active" : ""}`} to="/">
        Home
      </Link>
      <span className="crumb-sep">/</span>
      <Link className={`crumb ${location.pathname === "/results" ? "active" : ""}`} to="/results">
        Results
      </Link>
      <span className="crumb-sep">/</span>
      <Link className={`crumb ${location.pathname === "/about" ? "active" : ""}`} to="/about">
        About
      </Link>
    </nav>
  );
}

function AnimatedRoutes({ isOnline, apiStatus, onSaveResult, result, history }) {
  const location = useLocation();

  return (
    <div className="route-stage" key={location.pathname}>
      <Routes location={location}>
        <Route path="/" element={<HomePage isOnline={isOnline} apiStatus={apiStatus} onSaveResult={onSaveResult} />} />
        <Route path="/results" element={<ResultsPage result={result} history={history} />} />
        <Route path="/about" element={<AboutPage />} />
        <Route
          path="*"
          element={
            <section className="card page-card">
              <h2>Page Not Found</h2>
              <p className="placeholder">The page you requested does not exist.</p>
              <Link to="/" className="text-link">
                Go back to Home
              </Link>
            </section>
          }
        />
      </Routes>
    </div>
  );
}

function HomePage({ isOnline, apiStatus, onSaveResult }) {
  const [datasetRow, setDatasetRow] = useState(buildInitialDatasetRow);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const payloadFeatures = useMemo(() => {
    return FEATURE_FIELDS.reduce((acc, field) => {
      acc[field.key] = parseFeatureValue(datasetRow[field.key], 0);
      return acc;
    }, {});
  }, [datasetRow]);

  async function handleCsvUpload(file) {
    setError("");

    if (!file) {
      setDatasetRow(buildInitialDatasetRow);
      setUploadInfo(null);
      return;
    }

    try {
      const text = await readFileText(file);
      const row = parseCsvText(text);
      const { resolved, matchedColumns } = applyDatasetRow(row);

      setDatasetRow(resolved);
      setUploadInfo({
        fileName: file.name,
        matchedColumns,
        totalColumns: Object.keys(resolved).length
      });
    } catch (err) {
      setError(err.message || "Could not read the CSV file.");
    }
  }

  async function handlePredict(event) {
    event.preventDefault();

    if (!String(datasetRow.asn1 ?? "").trim() || !String(datasetRow.asn2 ?? "").trim()) {
      setError("Please upload a CSV row that contains both ASN values.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      // ===== LOCAL BACKEND =====
      // const response = await fetch(`${API_BASE}/predict`, {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({
      //     ispA: String(datasetRow.asn1).trim(),
      //     ispB: String(datasetRow.asn2).trim(),
      //     model: DEFAULT_MODEL_VALUE,
      //     features: payloadFeatures
      //   })
      // });
      // const data = await response.json();
      // const nextResult = {
      //   ispA: String(datasetRow.asn1).trim(),
      //   ispB: String(datasetRow.asn2).trim(),
      //   model: DEFAULT_MODEL_LABEL,
      //   label: data.label,
      //   probability: data.probability,
      //   createdAt: new Date().toISOString(),
      //   features: payloadFeatures
      // };

      // ===== HUGGING FACE (Gradio SSE v3) =====
      const ispA = String(datasetRow.asn1).trim();
      const ispB = String(datasetRow.asn2).trim();

      // Step 1: submit the job to Gradio named endpoint "predict"
      const startRes = await fetch(`${API_BASE}/gradio_api/call/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: [ispA, ispB, payloadFeatures] })
      });

      if (!startRes.ok) {
        const detail = await startRes.text().catch(() => null);
        throw new Error(detail ? `Gradio call failed: ${detail}` : `Gradio call failed with status ${startRes.status}.`);
      }

      const startJson = await startRes.json();
      const eventId = startJson?.event_id;
      if (!eventId) {
        throw new Error("Gradio did not return an event_id. The backend may be unavailable.");
      }

      // Step 2: read the SSE stream result
      const streamRes = await fetch(`${API_BASE}/gradio_api/call/predict/${eventId}`);

      if (!streamRes.ok) {
        const detail = await streamRes.text().catch(() => null);
        throw new Error(detail ? `Gradio stream failed: ${detail}` : `Gradio stream failed with status ${streamRes.status}.`);
      }

      const sseText = await streamRes.text();

      // Parse SSE lines and find the last payload containing output.data
      const dataLines = sseText
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice("data: ".length).trim())
        .filter(Boolean);

      let output = null;
      for (let i = dataLines.length - 1; i >= 0; i--) {
        try {
          const payload = JSON.parse(dataLines[i]);
          if (Array.isArray(payload?.output?.data)) {
            output = payload.output.data;
            break;
          }
        } catch {
          // skip non-JSON lines
        }
      }

      if (!output) {
        throw new Error("No valid output received from the Gradio backend. Please try again.");
      }

      const nextResult = {
        ispA,
        ispB,
        model: DEFAULT_MODEL_LABEL,
        label: output[0] ?? "Unknown",
        probability: Number(output[1] ?? 0) || 0,
        createdAt: new Date().toISOString(),
        features: output[2] ?? payloadFeatures
      };

      onSaveResult(nextResult);
      navigate("/results");
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid">
      <section className="card wide">
        <h2>Upload One Dataset Row</h2>
        <p className="form-note">
          Upload one CSV row that already contains <code>asn1</code>, <code>asn2</code>, <code>label</code>, and all pair features such as affinity score and customer cone values.
        </p>
        <form onSubmit={handlePredict}>
          <div className="row two-col">
            <label>
              Model
              <div className="model-display">{DEFAULT_MODEL_LABEL}</div>
            </label>
            <div className="upload-card upload-card-single">
              <span className="upload-label">Dataset CSV</span>
              <input type="file" accept=".csv,text/csv" onChange={(e) => handleCsvUpload(e.target.files?.[0] ?? null)} />
              <span className="upload-help">The first data row will be used. Missing feature values default to 0.</span>
              {uploadInfo ? (
                <span className="upload-status">
                  {uploadInfo.fileName} | {uploadInfo.matchedColumns} columns mapped
                </span>
              ) : (
                <span className="upload-status muted">No CSV loaded yet.</span>
              )}
            </div>
          </div>

          <section className="feature-summary">
            <h3>ASN Pair Preview</h3>
            <div className="meta-grid">
              {DATASET_META_FIELDS.map((key) => (
                <article key={key} className="feature-mini-card">
                  <span>{key.toUpperCase()}</span>
                  <strong>{String(datasetRow[key] ?? "") || "Missing"}</strong>
                </article>
              ))}
            </div>
          </section>

          <section className="feature-summary">
            <h3>Resolved Feature Snapshot</h3>
            <p className="form-note">
              The app sends every feature column from your dataset row to the API. Missing values are already filled with 0 before prediction.
            </p>
            <div className="feature-mini-grid full-grid">
              {FEATURE_FIELDS.map((field) => (
                <article key={field.key} className="feature-mini-card">
                  <span>{field.label}</span>
                  <strong>{payloadFeatures[field.key]}</strong>
                </article>
              ))}
            </div>
          </section>

          <p className="form-note">
            You do not need to type the individual feature values by hand anymore. The CSV should already contain both ASNs plus the pair features.
          </p>

          {error ? <p className="error">{error}</p> : null}

          <div className="row">
            <button type="submit" className="primary" disabled={loading || !isOnline || apiStatus.state === "offline"}>
              {loading ? "Processing..." : "Predict and Open Results"}
            </button>
            <Link className="text-link" to="/results">
              Go to Results Page
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}

function ResultsPage({ result, history }) {
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const latestResult = result || history[0] || null;

  async function handleCopy() {
    if (!latestResult) return;
    const summary = `${latestResult.ispA} <-> ${latestResult.ispB} | ${latestResult.label} (${Math.round(latestResult.probability * 100)}%)`;

    try {
      await navigator.clipboard.writeText(summary);
    } catch {
      setError("Unable to copy result to clipboard.");
    }
  }

  async function handleShare() {
    if (!latestResult || !navigator.share) return;

    try {
      await navigator.share({
        title: "Peering Prediction",
        text: `${latestResult.ispA} and ${latestResult.ispB}: ${latestResult.label} (${Math.round(latestResult.probability * 100)}%)`
      });
    } catch {
      // User canceled or share unavailable.
    }
  }

  return (
    <main className="grid">
      <section className="card">
        <h2>Prediction Result</h2>
        {!latestResult ? (
          <p className="placeholder">No prediction yet. Go to Home and submit a pair first.</p>
        ) : (
          <>
            <div className="result-block">
              <p>
                <strong>Pair:</strong> {latestResult.ispA} and {latestResult.ispB}
              </p>
              <p>
                <strong>Model:</strong> {latestResult.model}
              </p>
              <p>
                <strong>Label:</strong> {latestResult.label}
              </p>
              <p>
                <strong>Probability:</strong> {Math.round(latestResult.probability * 100)}%
              </p>
              <div className="meter-wrap">
                <div className="meter" style={{ width: `${Math.max(0, Math.min(100, latestResult.probability * 100))}%` }} />
              </div>
            </div>

            <div className="row">
              <button type="button" onClick={handleCopy} className="ghost">
                Copy Summary
              </button>
              {navigator.share ? (
                <button type="button" onClick={handleShare} className="ghost">
                  Share
                </button>
              ) : null}
            </div>
          </>
        )}
        {error ? <p className="error">{error}</p> : null}
      </section>

      <aside className="card sticky-summary">
        <h2>Last Predicted Pair</h2>
        {!latestResult ? (
          <p className="placeholder">No recent pair yet.</p>
        ) : (
          <>
            <p>
              <strong>{latestResult.ispA}</strong> and <strong>{latestResult.ispB}</strong>
            </p>
            <p>{latestResult.label}</p>
            <p>{Math.round(latestResult.probability * 100)}% confidence</p>
          </>
        )}
        <div className="row">
          <button type="button" className="ghost" onClick={() => navigate(-1)}>
            Go Back
          </button>
          <Link className="text-link" to="/">
            Home
          </Link>
        </div>
      </aside>

      <section className="card wide">
        <h2>Recent Predictions ({history.length})</h2>
        {history.length === 0 ? (
          <p className="placeholder">No saved predictions in localStorage yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item, index) => (
              <article key={`${item.createdAt}-${index}`} className="history-item">
                <p>
                  <strong>{item.ispA}</strong> and <strong>{item.ispB}</strong>
                </p>
                <p>
                  {item.label} | {Math.round(item.probability * 100)}% | {item.model}
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function AboutPage() {
  const navigate = useNavigate();

  return (
    <main className="grid">
      <section className="card wide page-card">
        <h2>About This Project</h2>
        <p>
          Peering Partner Prediction is a full-stack mini project that estimates whether two internet service providers are likely to establish a peering relationship.
          The goal is to turn raw ISP metadata into a simple binary decision that can help demonstrate how machine learning can support network planning.
        </p>
        <p>
          The frontend is built with React and Vite, while the backend is a Hugging Face Space API that loads the saved XGBoost model.
        </p>
        <div className="info-grid">
          <article className="info-card">
            <h3>Frontend</h3>
            <p>Vite build, static hosting, and <code>VITE_API_BASE_URL</code> pointed at the deployed API.</p>
          </article>
          <article className="info-card">
            <h3>Backend</h3>
            <p>FastAPI loads <code>optimal_pus_features.pkl</code> and exposes <code>/health</code> and <code>/predict</code>.</p>
          </article>
          <article className="info-card">
            <h3>Model files</h3>
            <p>Keep the saved model file in the Space repo's <code>models</code> folder, or set <code>MODEL_DIR</code>.</p>
          </article>
        </div>
        <div className="row">
          <button type="button" className="ghost" onClick={() => navigate(-1)}>
            Go Back
          </button>
          <Link className="text-link" to="/">
            Home
          </Link>
          <Link className="text-link" to="/results">
            Results
          </Link>
        </div>
      </section>
    </main>
  );
}
