import { Fragment } from "react";
import type { Prediction } from "../types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

function Bar({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <div className="cmp">
      <div className="cmp-label">{label}</div>
      <div className="cmp-row"><span className="tag clf">Classifier</span><div className="track"><div className="fill clf" style={{ width: pct(a) }} /></div><span className="num">{pct(a)}</span></div>
      <div className="cmp-row"><span className="tag hmm">+ HMM</span><div className="track"><div className="fill hmm" style={{ width: pct(b) }} /></div><span className="num">{pct(b)}</span></div>
    </div>
  );
}

export default function MetricsPanel({ p }: { p: Prediction }) {
  const m = p.metrics;
  const total = Object.values(p.timings_ms).reduce((s, v) => s + v, 0);
  return (
    <div className="panel">
      <h3>Trip {p.trip_id}</h3>
      <div className="kv">
        <span>GPS observations</span><b>{p.n_observations.toLocaleString()}</b>
        <span>Dropped (invalid)</span><b>{p.n_dropped_observations}</b>
        <span>Mean candidates / obs</span><b>{p.mean_candidates.toFixed(2)}</b>
        <span>Route arcs (reconstructed)</span><b>{p.route_arc_ids.length.toLocaleString()}</b>
        <span>Unresolved route breaks</span><b>{p.unresolved_breaks}</b>
        <span>Model</span><b className="mono">{p.model}</b>
      </div>
      <h4>Evaluation (computed server-side)</h4>
      {m ? (
        <>
          <div className="big">
            <div><div className="big-num">{pct(m.hmm_top1_accuracy)}</div><div className="big-lbl">Top-1 accuracy (HMM)</div></div>
            <div><div className="big-num">{pct(m.hmm_route_f1)}</div><div className="big-lbl">Route F1 (arcs)</div></div>
          </div>
          <Bar label="Observation Top-1 accuracy" a={m.classifier_top1_accuracy} b={m.hmm_top1_accuracy} />
          <Bar label="Arc macro F1" a={m.classifier_arc_macro_f1} b={m.hmm_arc_macro_f1} />
          <Bar label="Route continuity" a={m.classifier_route_continuity} b={m.hmm_route_continuity} />
          <div className="kv small">
            <span>Route precision / recall</span><b>{pct(m.hmm_route_precision)} / {pct(m.hmm_route_recall)}</b>
            <span>Candidate miss rate</span><b>{pct(m.candidate_miss_rate)}</b>
            <span>Evaluated observations</span><b>{m.n_evaluated_observations.toLocaleString()}</b>
          </div>
        </>
      ) : (
        <p className="muted">No server-side reference is available for this trip.</p>
      )}
      <h4>Pipeline timings</h4>
      <div className="kv small">
        {Object.entries(p.timings_ms).map(([k, v]) => (
          <Fragment key={k}><span>{k.replace(/_/g, " ")}</span><b>{v.toFixed(0)} ms</b></Fragment>
        ))}
        <span>total</span><b>{total.toFixed(0)} ms</b>
      </div>
      <p className="muted tiny">HMM: β={p.hmm_params.beta}, λ={p.hmm_params.lam}, unreachable log-p={p.hmm_params.unreachable_logp}</p>
    </div>
  );
}
