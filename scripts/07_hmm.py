"""HMM/Viterbi: tune decoder parameters on OUT-OF-FOLD probabilities of training trips,
then compare classifier-only vs classifier+HMM on the student-testing trips.

Usage: python scripts/07_hmm.py <model_tag>
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.evaluation import PROC, evaluate, load_partition, obs_predictions, trip_meta, unreliable_train_trips  # noqa: E402
from pathsnap.hmm import HMMParams, build_lattice, reconstruct_route, viterbi  # noqa: E402
from pathsnap.metrics import route_metrics  # noqa: E402
from pathsnap.network import ProjectedNetwork  # noqa: E402
from pathsnap.preprocessing import clean_track  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "hist_gbm_tuned"
RES = io.ROOT / "results"
GRID = dict(beta=[1.5, 3, 5, 10, 20, 40], lam=[0.15, 0.25, 0.35, 0.5, 1.0, 2.0], unreachable_logp=[-6.0, -12.0, -25.0, -50.0])


def trip_net(tid, part):
    ct = clean_track(io.load_track(tid, part))
    xy = ct.df[["x", "y"]].to_numpy()
    return ct, ProjectedNetwork(io.load_network(tid, part), ct.proj, bbox_xy=(xy.min(0), xy.max(0)))


def lattice_job(tid, part, cand):
    t0 = time.time()
    ct, pn = trip_net(tid, part)
    lat = build_lattice(ct.df, pn, cand)
    return tid, lat, time.time() - t0


def decode_all(lats, prm):
    return {tid: viterbi(lat, prm) for tid, lat in lats.items()}


def score_preds(O, preds):
    pred = np.concatenate([preds[t] for t in O.trip_id.unique()])
    return float(np.mean(pred == O.ref_arc.to_numpy())), pred


if __name__ == "__main__":
    # ---------- tuning on training trips (OOF scores, never fitted on the trip itself) ----------
    X, O = load_partition("train")
    bad = unreliable_train_trips()  # same exclusion as in 04_train_models.py, so rows align with the OOF file
    X, O = X[~X.trip_id.isin(bad)].reset_index(drop=True), O[~O.trip_id.isin(bad)].reset_index(drop=True)
    oof = pd.read_parquet(PROC / f"oof_{TAG}.parquet")["oof"].to_numpy()
    assert len(oof) == len(X) and not np.isnan(oof).any(), "OOF scores do not align with training rows"
    X["score"] = oof
    tune_trips = sorted(O.trip_id.unique())[::2]  # 39 trips, fixed deterministic subset
    O_t = O[O.trip_id.isin(tune_trips)].reset_index(drop=True)
    t0 = time.time()
    out = Parallel(n_jobs=12)(delayed(lattice_job)(t, "train", X[X.trip_id == t]) for t in tune_trips)
    lats = {t: lat for t, lat, _ in out}
    t_lat = time.time() - t0
    base_top1, _ = score_preds(O_t, {t: obs_predictions(X[X.trip_id == t], O_t[O_t.trip_id == t], "score").to_numpy() for t in tune_trips})
    trials = []
    for beta, lam, ul in itertools.product(*GRID.values()):
        prm = HMMParams(beta=beta, lam=lam, unreachable_logp=ul)
        t1 = time.time()
        acc, _ = score_preds(O_t, decode_all(lats, prm))
        trials.append(dict(beta=beta, lam=lam, unreachable_logp=ul, top1=acc, secs=time.time() - t1))
        print(f"beta={beta} lam={lam} unreach={ul} top1={acc:.4f}", flush=True)
    tr = pd.DataFrame(trials).sort_values("top1", ascending=False)
    tr.to_csv(RES / "hmm_tuning.csv", index=False)
    best = tr.iloc[0]
    prm = HMMParams(beta=float(best.beta), lam=float(best.lam), unreachable_logp=float(best.unreachable_logp))
    cfg = {"model_tag": TAG, "params": prm.__dict__, "tuning_trips": tune_trips, "tuning_top1": float(best.top1),
           "tuning_classifier_only_top1": base_top1, "lattice_build_s": t_lat, "grid": GRID}
    (io.ROOT / "artifacts" / "hmm_config.json").write_text(json.dumps(cfg, indent=1))
    print("best", cfg["params"], "tune top1", best.top1, "vs classifier", base_top1)

    # ---------- student-testing comparison ----------
    Xs, Os = load_partition("student_test")
    Xs["score"] = pd.read_parquet(PROC / f"st_{TAG}.parquet")["score"].to_numpy()
    rows, preds = [], {}
    for tid in Os.trip_id.unique():
        t0 = time.time()
        ct, pn = trip_net(tid, "student_test")
        cand = Xs[Xs.trip_id == tid]
        lat = build_lattice(ct.df, pn, cand)
        hp = viterbi(lat, prm)
        t_dec = time.time() - t0
        cp = obs_predictions(cand, Os[Os.trip_id == tid], "score").to_numpy()
        u, v, route = trip_meta(tid, "student_test")
        ref = Os[Os.trip_id == tid].ref_arc.to_numpy()
        mc, mh = route_metrics(cp, ref, route, u, v), route_metrics(hp, ref, route, u, v)
        xy = ct.df[["x", "y"]].to_numpy()
        rec_c, br_c = reconstruct_route(pn, cp, xy)
        rec_h, br_h = reconstruct_route(pn, hp, xy)
        def setf1(rec):
            s, r = set(rec), set(route.tolist()); i = len(s & r)
            p_, r_ = i / max(len(s), 1), i / max(len(r), 1)
            return 2 * p_ * r_ / max(p_ + r_, 1e-9)
        rows.append(dict(trip_id=tid, n_obs=len(ref), clf_top1=mc["top1"], hmm_top1=mh["top1"], clf_arc_f1=mc["arc_macro_f1"],
                         hmm_arc_f1=mh["arc_macro_f1"], clf_continuity=mc["continuity"], hmm_continuity=mh["continuity"],
                         clf_breaks=mc["n_breaks"], hmm_breaks=mh["n_breaks"], clf_route_f1=setf1(rec_c), hmm_route_f1=setf1(rec_h),
                         clf_unfilled=br_c, hmm_unfilled=br_h, decode_s=t_dec))
        preds[tid] = (cp, hp)
        print(rows[-1], flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(RES / "hmm_student_test_per_trip.csv", index=False)
    hp_all = pd.Series(np.concatenate([preds[t][1] for t in Os.trip_id.unique()]))
    res_c = evaluate(Xs, Os, "score", "student_test")
    res_h = evaluate(Xs, Os, None, "student_test", pred_obs=hp_all)
    summary = {"classifier_only": {k: v for k, v in res_c.items() if k != "confusion"},
               "classifier_hmm": res_h,
               "reconstructed_route_f1": {"classifier_only": float(per.clf_route_f1.mean()), "classifier_hmm": float(per.hmm_route_f1.mean())},
               "unfilled_breaks": {"classifier_only": int(per.clf_unfilled.sum()), "classifier_hmm": int(per.hmm_unfilled.sum())},
               "decode_s_total": float(per.decode_s.sum()), "hmm_params": prm.__dict__}
    (RES / "hmm_comparison.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez_compressed(PROC / "st_hmm_preds.npz", **{f"{t}_clf": preds[t][0] for t in preds}, **{f"{t}_hmm": preds[t][1] for t in preds})
    print(json.dumps(summary, indent=1, default=float))
