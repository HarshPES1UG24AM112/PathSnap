import { useCallback, useEffect, useState } from "react";
import { api, getApiBase, setApiBase } from "./api";
import MapView from "./components/MapView";
import MetricsPanel from "./components/MetricsPanel";
import type { Health, NetworkGeometry, Prediction, TripInfo } from "./types";

type Status = "idle" | "loading" | "done" | "error";

export default function App() {
  const [apiUrl, setApiUrl] = useState(getApiBase());
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);
  const [trips, setTrips] = useState<TripInfo[]>([]);
  const [partition, setPartition] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [network, setNetwork] = useState<NetworkGeometry | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const connect = useCallback(async () => {
    setHealthErr(null);
    setHealth(null);
    setTrips([]);
    try {
      const h = await api.health();
      setHealth(h);
      const t = await api.trips();
      setTrips(t.trips);
      setPartition(t.partition);
    } catch (e) {
      setHealthErr((e as Error).message);
    }
  }, []);

  useEffect(() => {
    connect();
  }, [connect]);

  const run = async (tripId: string) => {
    setSelected(tripId);
    setStatus("loading");
    setError(null);
    setPrediction(null);
    setNetwork(null);
    const t0 = performance.now();
    try {
      const p = await api.predict(tripId);
      setPrediction(p);
      setElapsed(performance.now() - t0);
      setStatus("done");
      api.network(tripId).then(setNetwork).catch(() => setNetwork(null));
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    }
  };

  return (
    <div className="app">
      <header>
        <div className="brand">
          <svg width="30" height="30" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#16365c" /><path d="M6 24 C12 20 12 10 18 10 S24 16 26 8" stroke="#ff5a4e" strokeWidth="3" fill="none" strokeLinecap="round" /><circle cx="6" cy="24" r="3" fill="#4ea1ff" /><circle cx="26" cy="8" r="3" fill="#4ea1ff" /></svg>
          <div>
            <div className="brand-name">PathSnap</div>
            <div className="brand-sub">Distributed map matching &amp; route reconstruction</div>
          </div>
        </div>
        <form
          className="api-form"
          onSubmit={(e) => {
            e.preventDefault();
            setApiBase(apiUrl);
            connect();
          }}
        >
          <label htmlFor="api">API (Laptop 1)</label>
          <input id="api" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} spellCheck={false} />
          <button type="submit">Connect</button>
          <span className={`pill ${health ? "ok" : healthErr ? "bad" : "wait"}`}>{health ? "API online" : healthErr ? "API offline" : "connecting…"}</span>
        </form>
      </header>

      <div className="overview">
        <div className="card"><div className="c-lbl">Evaluation trips</div><div className="c-val">{trips.length || "–"}</div><div className="c-sub">{partition || "—"}</div></div>
        <div className="card"><div className="c-lbl">Model</div><div className="c-val mono sm">{health?.model ?? "–"}</div><div className="c-sub">candidate scorer → HMM/Viterbi</div></div>
        <div className="card"><div className="c-lbl">Prediction status</div><div className={`c-val sm st-${status}`}>{{ idle: "Waiting for selection", loading: "Running pipeline…", done: "Completed", error: "Failed" }[status]}</div><div className="c-sub">{elapsed && status === "done" ? `round trip ${(elapsed / 1000).toFixed(2)} s` : selected ? `trip ${selected}` : ""}</div></div>
        <div className="card"><div className="c-lbl">Top-1 accuracy (HMM)</div><div className="c-val">{prediction?.metrics ? `${(prediction.metrics.hmm_top1_accuracy * 100).toFixed(1)}%` : "–"}</div><div className="c-sub">{prediction?.metrics ? `classifier only ${(prediction.metrics.classifier_top1_accuracy * 100).toFixed(1)}%` : "evaluated on the server"}</div></div>
      </div>

      {healthErr && (
        <div className="alert">
          <b>Cannot connect to the backend.</b> {healthErr}. Check that the API runs on Laptop 1 (<code>uvicorn backend.app.main:app --host 0.0.0.0 --port 8000</code>), that both laptops share a network, and that the URL above uses Laptop 1's IP.
        </div>
      )}

      <main>
        <aside className="trips">
          <h3>Select a trip</h3>
          {trips.length === 0 && !healthErr && <p className="muted">Loading trips…</p>}
          {trips.map((t) => (
            <button key={t.trip_id} className={`trip ${selected === t.trip_id ? "active" : ""}`} onClick={() => run(t.trip_id)} disabled={status === "loading"}>
              <div className="trip-id">Trip {t.trip_id}</div>
              <div className="trip-meta">{t.n_observations.toLocaleString()} GPS fixes · {(t.duration_s / 60).toFixed(1)} min</div>
              <div className="trip-meta">{t.n_arcs.toLocaleString()} road arcs</div>
            </button>
          ))}
        </aside>
        <section className="center">
          {status === "loading" && <div className="loading"><div className="spinner" />Matching trip {selected}: preprocessing → candidates → classifier → Viterbi…</div>}
          {status === "error" && <div className="alert">{error}</div>}
          <MapView prediction={prediction} network={network} />
        </section>
        <aside className="metrics">{prediction ? <MetricsPanel p={prediction} /> : <div className="panel muted">Trip details and evaluation metrics appear here after prediction.</div>}</aside>
      </main>
      <footer>Ground-truth reference routes stay on the API server; the UI only receives predictions and aggregate metrics.</footer>
    </div>
  );
}
