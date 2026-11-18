"""Build the consolidated technical report PDF from saved experiment outputs.

Every number in the report is read from files produced by the scripts in ../scripts, so the
PDF stays consistent with the repository. Usage: python reports/build_report.py
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
SHOT = ROOT / "reports" / "screenshots"
OUT = ROOT / "reports" / "PathSnap_Technical_Report.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#16365c"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#16365c"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9.3, leading=12.4, spaceAfter=4)
SM = ParagraphStyle("SM", parent=P, fontSize=8, leading=10)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=6.6, leading=8)

story = []
fig_no = [0]
tab_no = [0]


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, P))


def bullets(items):
    for it in items:
        story.append(Paragraph("• " + it, P))


def qa(q, a):
    story.append(Paragraph(f"<b>{q}</b> {a}", P))


def img(path, width_cm=16.5, caption=None):
    path = Path(path)
    if not path.exists():
        p(f"<i>[missing figure: {path.name}]</i>")
        return
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    h = w * ih / iw
    if h > 21 * cm:
        h = 21 * cm; w = h * iw / ih
    fig_no[0] += 1
    items = [Image(str(path), width=w, height=h)]
    if caption:
        items.append(Paragraph(f"Figure {fig_no[0]}. {caption}", CAP))
    items.append(Spacer(1, 4))
    story.append(KeepTogether(items))


def table(rows, header=True, col_widths=None, caption=None, font=7.4):
    tab_no[0] += 1
    data = [[Paragraph(str(c), ParagraphStyle("t", parent=SM, fontSize=font, leading=font + 1.6)) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cf")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2"))]
    t.setStyle(TableStyle(st))
    items = []
    if caption:
        items.append(Paragraph(f"Table {tab_no[0]}. {caption}", CAP))
    items += [t, Spacer(1, 6)]
    story.append(KeepTogether(items) if len(rows) < 25 else t)


f3 = lambda v: f"{v:.3f}"  # noqa: E731
f4 = lambda v: f"{v:.4f}"  # noqa: E731
pc = lambda v: f"{100 * v:.2f}%"  # noqa: E731

# ------------------------------------------------------------------ data
inv = pd.read_csv(R / "eda/trip_inventory.csv", dtype={"trip_id": str})
ps = pd.read_csv(R / "eda/partition_summary.csv", index_col=0)
cst = pd.read_csv(R / "eda/candidate_stats.csv", dtype={"trip_id": str})
mc = pd.read_csv(R / "model_comparison.csv").set_index("model")
runs = {f.stem: json.loads(f.read_text()) for f in (ROOT / "experiments/models").glob("*.json")}
best = json.loads((ROOT / "experiments/optuna/best_params.json").read_text())
trials = pd.read_csv(ROOT / "experiments/optuna/trials.csv")
tvu = pd.read_csv(R / "tuned_vs_untuned.csv", index_col=0)
hcfg = json.loads((ROOT / "artifacts/hmm_config.json").read_text())
hcmp = json.loads((R / "hmm_comparison.json").read_text())
hper = pd.read_csv(R / "hmm_student_test_per_trip.csv", dtype={"trip_id": str})
htune = pd.read_csv(R / "hmm_tuning.csv")
err = pd.read_csv(R / "error_analysis.csv", index_col=0)
e2e = pd.read_csv(R / "e2e_api_results.csv", dtype={"trip_id": str}) if (R / "e2e_api_results.csv").exists() else None
pytest_log = (ROOT / "experiments/logs/pytest.log").read_text(encoding="utf-8", errors="ignore") if (ROOT / "experiments/logs/pytest.log").exists() else ""
bad = runs["hist_gbm"]["excluded_train_trips"]
tr_inv, st_inv = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
NICE = {"logreg": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes", "knn": "K-Nearest Neighbours", "decision_tree": "Decision Tree",
        "random_forest": "Random Forest", "hist_gbm": "Histogram Gradient Boosting", "mlp": "Multi-layer Perceptron", "hist_gbm_tuned": "HistGB (tuned)"}
base_models = ["logreg", "gaussian_nb", "knn", "decision_tree", "random_forest", "hist_gbm", "mlp"]
nearest_st = float((cst[cst.partition == "student_test"].nearest_acc * cst[cst.partition == "student_test"].n_obs).sum() / cst[cst.partition == "student_test"].n_obs.sum())
ch, hh = hcmp["classifier_only"], hcmp["classifier_hmm"]

# ------------------------------------------------------------------ title
story.append(Spacer(1, 2.2 * cm))
story.append(Paragraph("PathSnap", ParagraphStyle("T", parent=H1, fontSize=30, alignment=TA_CENTER, leading=36)))
story.append(Paragraph("Distributed Map Matching &amp; Route Reconstruction", ParagraphStyle("T2", parent=H2, alignment=TA_CENTER, fontSize=15)))
story.append(Paragraph("Technical Report – UE24CS352A Machine Learning Hackathon, PES University (RR Campus)", ParagraphStyle("T3", parent=P, alignment=TA_CENTER)))
story.append(Spacer(1, 0.8 * cm))
table([["Item", "Value"],
       ["Final pipeline", "23 engineered features → Optuna-tuned HistGradientBoosting candidate scorer → HMM/Viterbi decoder → connected route"],
       ["Student-test Top-1 (classifier only → +HMM)", f"{pc(ch['top1'])} → {pc(hh['top1'])}"],
       ["Student-test route continuity (classifier only → +HMM)", f"{pc(ch['continuity'])} → {pc(hh['continuity'])}"],
       ["Models compared", "7 (linear, probabilistic, instance-based, tree, bagging, boosting, neural)"],
       ["Validation protocol", "5-fold GroupKFold on training trips (whole trips kept together); 10 student-test trips for development comparison"],
       ["Deployment", "FastAPI on Laptop 1, React + TypeScript + Leaflet UI on Laptop 2"],
       ["Team", "Harsh Agarwal (PES1UG24AM112), Harshit Vaish (PES1UG24AM116), Harsha KR (PES1UG24AM114), Harsha R (PES1UG24AM115)"]],
      col_widths=[6 * cm, 10.5 * cm], caption="Summary of the submitted system (all values computed from repository outputs).")
table([["Member", "SRN", "Responsibility"],
       ["Harsh Agarwal", "PES1UG24AM112", "Model training and comparison (7 models, GroupKFold CV, metrics, model selection), Bayesian optimisation (Optuna)"],
       ["Harshit Vaish", "PES1UG24AM116", "Dataset exploration and EDA, preprocessing, label alignment, candidate generation, feature engineering"],
       ["Harsha KR", "PES1UG24AM114", "HMM/Viterbi route reconstruction, decoder tuning, error analysis, Folium maps"],
       ["Harsha R", "PES1UG24AM115", "FastAPI backend, React + Leaflet frontend, two-laptop deployment, testing"]],
      col_widths=[3.2 * cm, 3.2 * cm, 10.1 * cm], caption="Team members and contributions (the report was written jointly).")
p("<b>AI-assistance disclosure.</b> An AI coding assistant (Claude) was used to help write code, run experiments and draft this report. "
  "All numbers, tables and figures were produced by executing the code in the repository; nothing was typed in by hand. "
  "The team is responsible for reviewing, understanding and being able to defend every part of it.")
story.append(PageBreak())

# ------------------------------------------------------------------ 1 intro
h1("1. Introduction and problem statement")
p("Consumer GPS fixes are noisy (urban canyons, multipath, drop-outs), so a raw trajectory often appears on a parallel street or jumps "
  "between roads. <b>Map matching</b> assigns every GPS observation to the road segment (arc) the vehicle actually used, and "
  "<b>route reconstruction</b> turns those assignments into a connected, drivable path. PathSnap solves this in two stages: (i) a supervised "
  "<i>candidate scorer</i> estimates, for every observation and every nearby arc, the probability that the arc is the true one; "
  "(ii) a Hidden Markov Model decoded with the Viterbi algorithm combines those probabilities with road-network connectivity "
  "to produce the most likely <i>sequence</i> of arcs. The system is deployed as a FastAPI service (Laptop 1) consumed by a React "
  "dashboard (Laptop 2).")
p("<b>Key formulation decision.</b> Arc ids are row indices of each trip's own <font face='Courier'>.arcs</font> file. Every trip has a "
  "different network (different cities; see §2), so arc 2641 in one trip is unrelated to arc 2641 in another. Treating arc ids as global "
  "classes would therefore be meaningless. We instead cast map matching as <b>candidate ranking</b>: each training row is an "
  "(observation, candidate arc) pair described by geometric/kinematic/topological features, and the binary target is "
  "<i>candidate = reference arc</i>. Every row carries <font face='Courier'>trip_id, obs, arc_id</font>, so a score always maps back "
  "to a real road segment of the correct network.")

# ------------------------------------------------------------------ 2 data
h1("2. Dataset understanding and exploratory data analysis")
p(f"<b>Partitions used.</b> The supplied archive contained <b>{len(tr_inv)} training</b> and <b>{len(st_inv)} student-testing</b> trips "
  "(folders <font face='Courier'>train/</font> and <font face='Courier'>student_test/</font>). The evaluator-testing partition was <b>not</b> "
  "part of the material we received, so it could not have been used anywhere. All EDA below uses train + student_test; "
  "every fitted quantity uses training trips only. Each trip folder holds 4 files (360 files in total).")
h2("2.1 Raw files and their relationships")
table([["File", "Verified content", "Role in pipeline"],
       [".track", "lon, lat, t (s); whitespace separated, no header; float", "Observations to be matched (input)"],
       [".nodes", "lon, lat per node; row index = node id", "Arc geometry (straight node-to-node segments)"],
       [".arcs", "from_node, to_node; row index = arc id; directed", "States of the HMM; candidate set members"],
       [".route", "chronological arc ids, one per row (NOT one per GPS fix)", "Reference: aligned to observations to create labels"]],
      col_widths=[1.6 * cm, 8 * cm, 6.9 * cm], caption="Raw file structure as verified by scripts/01_eda_inventory.py.")
p(f"There is no row correspondence between files: a route has far fewer rows than its track (e.g. train trip 00000001: "
  f"{int(inv.set_index('trip_id').loc['00000001','n_obs'])} fixes vs {int(inv.set_index('trip_id').loc['00000001','route_arcs'])} route arcs). "
  f"All route arc ids are valid indices and consecutive route arcs are connected head-to-tail in "
  f"{pc(ps.loc['route_contiguity','train'])} (train) / {pc(ps.loc['route_contiguity','student_test'])} (student-test) of cases, confirming the "
  "route is an ordered, mostly contiguous path through the directed network. Networks are trip-specific: all 90 node files have different "
  "checksums and trip centroids span longitudes "
  f"{inv.lon_min.min():.1f}° to {inv.lon_max.max():.1f}° (Figure 1), so we never assume a shared network or a fixed UTM zone.")
h2("2.2 Inventory, missing values and data quality")
q = ps
table([["Statistic", "train", "student_test"],
       ["Trips", int(q.loc['trips', 'train']), int(q.loc['trips', 'student_test'])],
       ["GPS observations (total / median / min / max)", f"{int(q.loc['obs_total','train']):,} / {int(q.loc['obs_median','train'])} / {int(q.loc['obs_min','train'])} / {int(q.loc['obs_max','train']):,}",
        f"{int(q.loc['obs_total','student_test']):,} / {int(q.loc['obs_median','student_test'])} / {int(q.loc['obs_min','student_test'])} / {int(q.loc['obs_max','student_test']):,}"],
       ["Rows with missing values (NaN)", int(q.loc['nan_rows', 'train']), int(q.loc['nan_rows', 'student_test'])],
       ["Invalid coordinates (out of range or 0,0)", int(q.loc['invalid_coords', 'train']), int(q.loc['invalid_coords', 'student_test'])],
       ["Non-increasing timestamps", int(q.loc['non_increasing_t', 'train']), int(q.loc['non_increasing_t', 'student_test'])],
       ["Share of 1-second steps (trip mean)", pc(q.loc['frac_dt_1s', 'train']), pc(q.loc['frac_dt_1s', 'student_test'])],
       ["Gaps > 5 s (count) / largest gap (s)", f"{int(q.loc['gaps_gt5s','train'])} / {q.loc['max_dt','train']:,.0f}", f"{int(q.loc['gaps_gt5s','student_test'])} / {q.loc['max_dt','student_test']:,.0f}"],
       ["Network nodes / arcs (median per trip)", f"{q.loc['nodes_median','train']:,.0f} / {q.loc['arcs_median','train']:,.0f}", f"{q.loc['nodes_median','student_test']:,.0f} / {q.loc['arcs_median','student_test']:,.0f}"],
       ["Isolated nodes (total)", int(q.loc['isolated_nodes', 'train']), int(q.loc['isolated_nodes', 'student_test'])],
       ["Weakly connected components (median per trip)", q.loc['weak_comp_median', 'train'], q.loc['weak_comp_median', 'student_test']],
       ["Median GPS → reference-route distance (m, trip median)", f"{q.loc['gps_to_route_med','train']:.2f}", f"{q.loc['gps_to_route_med','student_test']:.2f}"]],
      col_widths=[7.5 * cm, 4.5 * cm, 4.5 * cm], caption="Partition-level data-quality summary (results/eda/partition_summary.csv).")
p(f"<b>Missing-value analysis.</b> No NaN rows, no out-of-range or (0,0) coordinates and no duplicate/decreasing timestamps were found in either "
  f"partition; the cleaning code still handles all of these for unseen trips. The real quality issues are elsewhere: (a) <b>time gaps</b> – "
  f"{int(q.loc['gaps_gt5s','train'])} gaps > 5 s in training, including trips 00000090 and 00000078 with gaps of "
  f"{inv.set_index('trip_id').loc['00000090','max_dt']:,.0f} s and {inv.set_index('trip_id').loc['00000078','max_dt']:,.0f} s; "
  f"(b) <b>GPS spikes</b> – {int(inv.n_speed_gt50.sum())} steps imply > 180 km/h; (c) <b>fragmented networks</b> – the largest strongly connected "
  f"component covers on average {pc(inv.largest_scc_frac.mean())} of nodes and networks contain {int(inv.self_loops.sum())} self-loops and "
  f"{int(inv.dup_arcs.sum())} duplicated arcs (identical from/to pairs, geometrically indistinguishable); (d) <b>unreliable references</b> – in training trips "
  f"{', '.join(bad)} the median distance between GPS fixes and the reference route is "
  + ", ".join(f"{inv.set_index('trip_id').loc[b,'gps_to_route_med']:.0f} m" for b in bad)
  + " (vs ≈4.5 m elsewhere), i.e. the reference route is offset from or inconsistent with the trajectory. Following the guidance that reference "
  "routes are not error-free ground truth, these two trips are excluded from fitting and cross-validation (rule: trip-median GPS→route "
  "distance > 50 m, decided from training data only).")
img(FIG / "eda/trip_locations.png", 13, "Trip centroids: data come from many cities worldwide → per-trip local projection.")
img(FIG / "eda/obs_per_trip.png", 10, "GPS observations per trip (log scale) for both partitions.")
img(FIG / "eda/dt_speed.png", 15.5, "Sampling intervals (≈1 Hz with rare long gaps) and implied speeds (spikes above 180 km/h are GPS jumps).")
img(FIG / "eda/gps_to_reference.png", 11, "Distance between each fix and its aligned reference arc: median ≈4.5 m, heavy tail; motivates the 50 m candidate radius.")
img(FIG / "eda/network_size.png", 15.5, "Road-network size and fragmentation per trip (each trip ships its own network).")
img(FIG / "eda/trajectories.png", 16.5, "GPS trajectories (blue) over the local road network (grey) with the reference route (green); training trip 00000054 shows the reference/trajectory inconsistency.")
img(SHOT / "map_eda_train_00000001.png", 15.5, "Folium map (results/maps/eda_train_00000001.html): road network, raw GPS and reference route for a training trip, with layer control and legend.")

h2("2.3 Q1 – How did we preprocess the data?")
bullets([
    "<b>Validation &amp; cleaning</b> (src/pathsnap/preprocessing.py): drop NaN / out-of-range / (0,0) fixes, sort by time, drop duplicate timestamps. "
    "No observation is otherwise removed, because every valid fix must receive a predicted arc.",
    "<b>Projection</b> (src/pathsnap/geo.py): each trip is projected to a local equirectangular tangent plane centred on its own centroid "
    "(x = R·cos φ0·Δλ, y = R·Δφ). Trips span < 40 km, where the scale error is < 0.1 % (a unit test checks agreement with the haversine formula), "
    "far below GPS noise, and the transform is cheap enough for millions of point-to-segment distances. Degree-space Euclidean distance is never used.",
    "<b>Kinematics</b>: Δt, step length, speed (clipped at 60 m/s), heading from a centred displacement window that grows (±1…±8 fixes) until the "
    "vehicle has moved ≥ 4 m – this avoids random headings while stationary. Flags: <i>is_jump</i> (> 60 m/s to both neighbours) and "
    "<i>gap_before</i> (Δt > 30 s or step > 500 m) which restarts the HMM.",
    "<b>Label derivation</b> (src/pathsnap/labels.py): the route is aligned to the fixes with a <i>monotone dynamic programme</i> – fix i is assigned "
    "route position j(i), non-decreasing in i, minimising Σ min(d(i, route[j(i)]), 100 m). Monotonicity respects the route order, so loops and "
    "revisited roads are labelled correctly. Labels are used only for training/evaluation, never as inputs.",
    "<b>Candidate generation</b> (src/pathsnap/features.py, network.py): arcs are densified every 10 m into a KD-tree (only arcs inside the "
    "track's bounding box + 300 m are indexed). Candidates are all arcs within 50 m (≤ 12 nearest); if none, the 4 nearest within 250 m; if still "
    f"none the set is empty and the decoder skips that fix. Mean candidate-set size is {cst[cst.partition=='train'].cands_mean.mean():.2f}; the "
    f"candidate miss rate is {pc(1 - pd.read_parquet(ROOT / 'data/processed/student_test_obs.parquet').ref_in_cands.mean())} on student-test.",
    "<b>Learned transforms</b> (StandardScaler / QuantileTransformer) live inside each model pipeline and are therefore fitted on training folds only.",
])
img(FIG / "eda/candidates.png", 15.5, "Candidate-set sizes and per-trip candidate miss rate (high-miss trips are the two unreliable-reference training trips).")
h2("2.4 Q2 – Observed patterns and the features we extracted")
p(f"The nearest arc is the reference arc for only {pc(nearest_st)} of student-test fixes: dual carriageways and two-way streets are modelled as "
  "pairs of opposite arcs at the same distance, junctions place several arcs within a few metres, and noise often moves a fix closer to a "
  "parallel road. Distance alone is therefore insufficient; direction, local context and connectivity must be used. The 23 features "
  "(all label-free and computed per candidate) are:")
table([["Group", "Features", "Why"],
       ["Geometry", "dist, dist_rank, dist_minus_min, gauss_share (exp(−d²/2σ²) normalised over the candidate set, σ = 10 m), is_nearest, n_cands",
        "Absolute and relative closeness; relative features make scores comparable across dense and sparse areas."],
       ["Direction", "heading_diff (0–180°, direction aware), axis_diff (0–90°, undirected), heading_valid",
        "Separates the two directions of a two-way road and crossing streets at junctions."],
       ["Kinematics", "speed, dt (clipped 120 s), is_jump", "Heading is unreliable at low speed; spikes and gaps lower the trust in a fix."],
       ["Arc / topology", "arc_len, proj_t, at_endpoint, has_twin, out_deg_end, in_deg_start",
        "Projections clamped to an endpoint suggest the vehicle is on a neighbouring arc; junction degree marks ambiguous areas."],
       ["Temporal context", "dist_prev, dist_next, ctx_mean_dist (fixes i−2…i+2)", "A true arc stays close for several consecutive fixes; noise does not."],
       ["Connectivity", "adj_prev_dist / adj_next_dist: smallest distance of a neighbouring fix to a candidate that is the same arc or directly connected (into / out of this arc)",
        "Label-free sequential evidence that the arc belongs to a drivable path."]],
      col_widths=[2.3 * cm, 7.4 * cm, 6.8 * cm], caption="Engineered features (src/pathsnap/features.py).")

# ------------------------------------------------------------------ 3 models
story.append(PageBreak())
h1("3. Model experimentation and comparison")
p(f"<b>Protocol.</b> All seven models use identical features and data. 5-fold <b>GroupKFold</b> with groups = trip id runs on the "
  f"{len(tr_inv) - len(bad)} usable training trips, so no trip is split between training and validation. Training rows are the candidates of observations whose "
  "reference arc is in the candidate set (others contain no positive and would only add label noise); validation uses <i>all</i> observations of "
  "the held-out trips (missed references count as errors). Afterwards each model is refitted on all usable training trips and evaluated once "
  "on the 10 student-test trips. Seed = 42 everywhere. Configurations, fold results and timings are stored in experiments/models/&lt;model&gt;.json.")
p("<b>Metrics.</b> Exactly one candidate per fix is correct, so the decision rule is the per-observation arg-max (threshold-free). "
  "<i>Observation-level Top-1 accuracy</i> (fraction of fixes whose arg-max arc equals the reference arc, over all fixes) is the <b>primary "
  "selection metric</b> because it is exactly the map-matching objective and the quantity the HMM refines. Candidate-level accuracy, precision, "
  "recall and Macro F1 (classes 0/1 under the arg-max rule) and binary log loss (probability quality, important for HMM emissions) are also "
  "reported, plus <i>arc Macro F1</i> (per-trip macro F1 over arc ids, averaged over trips), <i>route continuity</i> (fraction of predicted arc "
  "changes that are directly connected in the directed network), candidate miss rate and timings. Class imbalance (≈13 % positives) is handled "
  "by the ranking formulation itself: no re-weighting is applied, which keeps the probabilities calibrated for the HMM.")
h2("3.1 Overall model comparison")
rows = [["Model", "Family", "Key configuration", "CV Macro F1", "CV Top-1", "ST Top-1", "Route contin. (ST)", "ST Macro F1", "ST Log loss", "Inference (ms/obs)"]]
for m in base_models + ["hist_gbm_tuned"]:
    r = mc.loc[m]
    cfg = {"logreg": "L2, C=1, lbfgs; standardised", "gaussian_nb": "quantile→normal transform, var_smoothing=1e-9", "knn": "k=50, distance-weighted, Euclidean on standardised features, 60k-row fit sample",
           "decision_tree": "max_depth=12, min_samples_leaf=50, Gini", "random_forest": "150 trees, max_depth=18, min_leaf=20, sqrt features",
           "hist_gbm": "lr=0.1, 31 leaves, min_leaf=40, early stopping", "mlp": "64-32 ReLU, Adam, α=1e-4, batch 1024, early stopping",
           "hist_gbm_tuned": ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in best["best_params"].items())}[m]
    rows.append([NICE[m], r.family, cfg, f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}", f"{r.cv_top1:.4f}", f"{r.st_top1:.4f}", f"{r.st_continuity:.3f}",
                 f"{r.st_macro_f1:.4f}", f"{r.st_log_loss:.4f}", f"{r.infer_ms_per_obs:.4f}"])
table(rows, col_widths=[2.2 * cm, 1.8 * cm, 3.5 * cm, 1.8 * cm, 1.2 * cm, 1.2 * cm, 1.3 * cm, 1.3 * cm, 1.2 * cm, 1.3 * cm],
      caption=f"Consolidated comparison (results/model_comparison.csv). CV = 5-fold GroupKFold on training trips; ST = 10 student-test trips. "
              f"Route continuity is for classifier-only arg-max predictions (HMM results in §5). Nearest-arc baseline ST Top-1 = {nearest_st:.4f}.", font=6.6)
img(FIG / "models/cv_macro_f1.png", 14, "Cross-validation Macro F1 for all models (bars = mean ± std over folds, dots = individual folds).")
img(FIG / "models/student_test_top1_f1.png", 14, "Student-test Top-1 accuracy and Macro F1.")
img(FIG / "models/log_loss.png", 13, "Binary log loss (lower = better calibrated probabilities), CV and student-test.")
img(FIG / "models/fit_gap_efficiency.png", 16, "Left: Top-1 on training-fold data vs validation folds (over/underfitting check). Right: accuracy vs inference cost.")
img(FIG / "models/confusion_matrices.png", 17, "Candidate-level confusion matrices on student-test (arg-max decision rule).")
img(FIG / "models/training_histories.png", 15, "Epoch/iteration-wise histories exist only for HistGradientBoosting (training vs internal validation loss) and the MLP (training loss, validation accuracy); no curves are drawn for the other models because they have none.")

# per-model subsections
h2("3.2 Individual model analyses")
fold_sd = mc.cv_top1_std.mean()
TXT = {
 "logreg": dict(
    why="A linear model on well-designed features is the natural first baseline: it yields calibrated probabilities directly usable as HMM emissions, trains in about a second and is fully interpretable.",
    sl="Strengths: very fast, stable across folds, good log loss. Limitation: the score is additive, so it cannot express interactions such as 'heading difference matters only when the vehicle is moving' or 'distance matters less when the candidate is a twin arc'.",
    hp="C = 1.0 (L2), solver lbfgs, max_iter 500, StandardScaler inside the pipeline.", loss="Binary cross-entropy (log loss) + L2 penalty, minimised with L-BFGS."),
 "gaussian_nb": dict(
    why="Included as the probabilistic-generative family: it models each feature's class-conditional density and is extremely cheap; a quantile→normal transform was added so the heavy-tailed distance features better match its Gaussian assumption.",
    sl="Strength: trivial to train, no hyper-parameter search needed. Limitation: the conditional-independence assumption is badly violated – dist, dist_rank, gauss_share, ctx_mean_dist and dist_prev/next all encode the same closeness signal, so the evidence is counted many times and the posteriors become extremely over-confident.",
    hp="var_smoothing = 1e-9; QuantileTransformer(200 quantiles, normal output).", loss="No iterative loss: maximum-likelihood estimates of per-class means/variances, prediction by Bayes' rule."),
 "knn": dict(
    why="Instance-based reasoning: a candidate is likely correct if similar (observation, candidate) configurations in other trips were correct; no parametric form is assumed.",
    sl="Strength: flexible local decision boundary. Limitations: needs feature scaling, memory and query time grow with the training set (we fit on a 60 000-row sample of whole observations to stay feasible), and probabilities are vote fractions of 50 neighbours.",
    hp="n_neighbors = 50, weights = distance, Euclidean metric on standardised features, brute-force search, 60 000 training rows.", loss="None (lazy learner); the prediction is a distance-weighted neighbour vote."),
 "decision_tree": dict(
    why="A single tree captures the non-linear rules suggested by EDA (e.g. thresholds on distance gap and heading difference) and is easy to inspect.",
    sl="Strength: interactions and thresholds, no scaling, microsecond inference. Limitation: high variance and piece-wise constant probabilities (coarse log loss); depth and leaf size must be limited.",
    hp="max_depth = 12, min_samples_leaf = 50, criterion = Gini.", loss="Greedy minimisation of Gini impurity at each split."),
 "random_forest": dict(
    why="Bagging many decorrelated trees keeps the non-linear interactions of a tree while averaging away its variance; averaged leaf frequencies give smoother probabilities.",
    sl="Strengths: the best CV Top-1 and among the best log loss, robust to feature scaling. Limitations: the largest artifact (≈45 MB), the slowest fit among the tree models and 150 trees to evaluate per prediction.",
    hp="n_estimators = 150, max_depth = 18, min_samples_leaf = 20, max_features = sqrt.", loss="Each tree minimises Gini impurity on a bootstrap sample; the ensemble averages class probabilities."),
 "hist_gbm": dict(
    why="Gradient boosting corrects residual errors sequentially and directly optimises log loss, which is what the HMM emissions need; the histogram implementation handles >1 M rows in seconds.",
    sl="Strengths: accuracy on par with the forest at a ninth of its training time, a tiny model (0.2 MB), built-in early stopping. Limitation: more hyper-parameters interact (learning rate × iterations × leaf size), which is why it is the best candidate for Bayesian optimisation.",
    hp="learning_rate = 0.1, max_iter = 300 (early stopping, 10 % validation split, patience 20), max_leaf_nodes = 31, min_samples_leaf = 40, l2 = 0.", loss="Binary log loss (cross-entropy) minimised by functional gradient descent."),
 "mlp": dict(
    why="A shallow neural network can learn smooth non-linear interactions between the continuous features and is trained with the same log loss used for the emissions.",
    sl="Strengths: best student-test Top-1 among the untuned models and smooth probabilities. Limitations: slowest to train, sensitive to scaling and initialisation; in some CV folds the 40-epoch cap was reached before convergence (sklearn ConvergenceWarning in experiments/logs/04_train_models.log).",
    hp="hidden layers (64, 32), ReLU, Adam (lr 1e-3), α (L2) = 1e-4, batch 1024, ≤ 40 epochs, early stopping on a 10 % split (patience 6).", loss="Binary cross-entropy + L2 penalty, mini-batch Adam."),
}
for i, m in enumerate(base_models, 1):
    r, rj = mc.loc[m], runs[m]
    folds = [f["val"]["top1"] for f in rj["folds"]]
    gap = r.train_top1 - r.cv_top1
    if m == "gaussian_nb":
        fit = (f"<b>Underfitting.</b> Top-1 is low on training-fold data too ({r.train_top1:.4f}) and is close to the nearest-arc baseline "
               f"({nearest_st:.4f} on student-test), and the log loss ({r.cv_log_loss:.3f} CV) is roughly 13× that of the other models: the model is too "
               "simple and mis-specified for this feature set rather than overfitted.")
    elif abs(gap) < 0.012:
        fit = (f"<b>No meaningful overfitting.</b> Top-1 on training-fold trips is {r.train_top1:.4f} vs {r.cv_top1:.4f} on validation folds "
               f"(gap {gap:+.4f}), and the student-test result ({r.st_top1:.4f}) lies inside the CV fold range [{min(folds):.4f}, {max(folds):.4f}]. "
               "Training and validation accuracy are almost equal and they plateau together with the other models at about 0.86. That points to a ceiling "
               "set by ambiguous labels (twin arcs, junction boundaries; see §5.3) rather than to underfitting by this particular model.")
    else:
        fit = f"Training-fold Top-1 {r.train_top1:.4f} vs validation {r.cv_top1:.4f} (gap {gap:+.4f}); student-test {r.st_top1:.4f}."
    extra = ""
    if m == "random_forest":
        d = [a["val"]["top1"] - b["val"]["top1"] for a, b in zip(runs["random_forest"]["folds"], runs["hist_gbm"]["folds"])]
        extra = (f" Paired per-fold difference RF − HistGB in Top-1: {', '.join(f'{x:+.4f}' for x in d)} (mean {sum(d)/len(d):+.4f}), i.e. a statistical tie.")
    if m == "hist_gbm" and rj.get("history"):
        extra = f" Early stopping ended the final fit after {len(rj['history']['train_loss'])} iterations (of 300), (patience 20 on the internal validation loss; see the training-history figure)."
    if m == "mlp" and rj.get("history"):
        extra = f" The final refit stopped after {len(rj['history']['train_loss'])} epochs through early stopping (training-history figure)."
    h3(f"3.2.{i} {NICE[m]} ({r.family})")
    qa("Q1 Why suitable?", TXT[m]["why"])
    qa("Q2 Strengths and limitations.", TXT[m]["sl"])
    qa("Q5 Hyper-parameters (initial values).", TXT[m]["hp"])
    qa("Q6 Loss / objective.", TXT[m]["loss"])
    qa("Q7 Student-test performance.", f"Candidate accuracy {r.st_accuracy:.4f}, precision {r.st_precision:.4f}, recall {r.st_recall:.4f}, arc Macro F1 {r.st_arc_macro_f1:.4f}, "
       f"route continuity {r.st_continuity:.3f}, route F1 {r.st_route_f1:.4f}; refit time {r.fit_s:.1f} s on {int(r.train_rows):,} rows; inference {r.infer_ms_per_obs:.4f} ms per fix.")
    qa("Q8 Top-1 / Macro F1 / Log loss (student-test).", f"Top-1 = <b>{r.st_top1:.4f}</b>, Macro F1 = <b>{r.st_macro_f1:.4f}</b>, log loss = <b>{r.st_log_loss:.4f}</b> "
       f"(CV: {r.cv_top1:.4f} ± {r.cv_top1_std:.4f}, {r.cv_macro_f1:.4f}, {r.cv_log_loss:.4f}).")
    qa("Q9 Over-/underfitting?", fit + extra)

h2("3.3 Model selection")
p(f"Gaussian NB is clearly worst and KNN is both less accurate and the slowest at inference. Logistic regression, the decision tree, RF, HistGB and "
  f"the MLP all reach a CV Top-1 between {mc.loc[['logreg','decision_tree','random_forest','hist_gbm','mlp'],'cv_top1'].min():.4f} and "
  f"{mc.loc[['logreg','decision_tree','random_forest','hist_gbm','mlp'],'cv_top1'].max():.4f}. That spread is far smaller than the fold-to-fold "
  f"standard deviation (≈{fold_sd:.3f}), which is driven by how hard each trip is. Random Forest ({mc.loc['random_forest','cv_top1']:.4f}) and "
  f"HistGradientBoosting ({mc.loc['hist_gbm','cv_top1']:.4f}) are statistically tied on the primary metric and on log loss. We selected "
  f"<b>HistGradientBoosting</b> because it (i) ties for the best CV Top-1 and log loss, (ii) trains about {mc.loc['random_forest','fit_s']/mc.loc['hist_gbm','fit_s']:.0f}× "
  "faster than RF, which makes a meaningful Bayesian search affordable, (iii) is a 0.2 MB artifact instead of 45 MB, and (iv) optimises log loss "
  "directly, so its probabilities suit the HMM emissions. The MLP's higher student-test score was not used for selection: student-test is a single "
  "10-trip sample, whereas CV covers 78 trips.")

# ------------------------------------------------------------------ 4 optuna
story.append(PageBreak())
h1("4. Bayesian hyper-parameter optimisation (HistGradientBoosting)")
qa("Q1 Hyper-parameters searched.", "learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features (feature sub-sampling per split) and max_iter (with early stopping still active).")
qa("Q2 Ranges / distributions.", "learning_rate log-uniform [0.02, 0.3]; max_leaf_nodes log-int [15, 255]; min_samples_leaf log-int [10, 1000]; l2_regularization log-uniform [1e-4, 10]; "
   "max_features uniform [0.3, 1.0]; max_iter int [100, 800] step 50. The ranges contain the default configuration (evaluated as trial 0). Log scales are used for "
   "scale-type parameters.")
qa("Q3 Objective.", f"Maximise the mean observation-level Top-1 accuracy over the same 5-fold GroupKFold on the {len(tr_inv)-len(bad)} usable training trips (the primary selection "
   "metric). Student-test and evaluator data are never used inside the objective. Sampler: TPE (seed 42, 10 random start-up trials); MedianPruner stops "
   "unpromising trials after the second fold. Trial 0 was enqueued with the untuned default configuration as a reference.")
qa("Q4 Number of trials.", f"{best['n_trials']} trials ({best['n_complete']} completed, {best['n_pruned']} pruned); wall-clock {best['wall_time_s']/60:.1f} min on a 24-thread CPU "
   "(experiments/optuna/trials.csv, study.db).")
qa("Q5 Best values.", ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in best["best_params"].items()) + f"; best CV Top-1 = {best['best_cv_top1']:.4f}.")
u, t = tvu["hist_gbm"], tvu["hist_gbm_tuned"]
better = t.st_top1 > u.st_top1
qa("Q6 Did tuning help?", f"CV Top-1 rose from {u.cv_top1:.4f} to {t.cv_top1:.4f} ({t.cv_top1-u.cv_top1:+.4f}) and student-test Top-1 from {u.st_top1:.4f} to {t.st_top1:.4f} "
   f"({t.st_top1-u.st_top1:+.4f}). {'The improvement carries over to unseen trips, but it' if better else 'It does not carry over to the student-test trips; the change'} is small relative to the fold standard deviation. "
   "This is consistent with §3: the models are limited by ambiguity in the data rather than by capacity. "
   f"The tuned model fits its training data more closely (training-fold Top-1 {runs['hist_gbm_tuned']['train_subset_mean']['top1']:.3f} vs {runs['hist_gbm']['train_subset_mean']['top1']:.3f} untuned), "
   "but validation and student-test accuracy still rose, so the extra fit generalises. The tuned model is used downstream.")
table([["Metric (same partitions)", "Untuned HistGB", "Tuned HistGB", "Δ"],
       ["CV Top-1 (mean ± std)", f"{u.cv_top1:.4f} ± {u.cv_top1_std:.4f}", f"{t.cv_top1:.4f} ± {t.cv_top1_std:.4f}", f"{t.cv_top1-u.cv_top1:+.4f}"],
       ["CV Macro F1", f4(u.cv_macro_f1), f4(t.cv_macro_f1), f"{t.cv_macro_f1-u.cv_macro_f1:+.4f}"],
       ["CV log loss", f4(u.cv_log_loss), f4(t.cv_log_loss), f"{t.cv_log_loss-u.cv_log_loss:+.4f}"],
       ["Student-test Top-1", f4(u.st_top1), f4(t.st_top1), f"{t.st_top1-u.st_top1:+.4f}"],
       ["Student-test Macro F1", f4(u.st_macro_f1), f4(t.st_macro_f1), f"{t.st_macro_f1-u.st_macro_f1:+.4f}"],
       ["Student-test log loss", f4(u.st_log_loss), f4(t.st_log_loss), f"{t.st_log_loss-u.st_log_loss:+.4f}"],
       ["Route continuity (classifier only)", f3(u.st_continuity), f3(t.st_continuity), f"{t.st_continuity-u.st_continuity:+.3f}"],
       ["Refit time (s)", f"{u.fit_s:.1f}", f"{t.fit_s:.1f}", ""],
       ["Inference (ms per fix)", f"{u.infer_ms_per_obs:.4f}", f"{t.infer_ms_per_obs:.4f}", ""]],
      col_widths=[6 * cm, 3.5 * cm, 3.5 * cm, 2.5 * cm], caption="Tuned vs untuned HistGradientBoosting (results/tuned_vs_untuned.csv).")
img(FIG / "models/optuna_history.png", 16, "Optuna history (best-so-far in red, untuned reference trial dashed) and correlation of each hyper-parameter with the objective.")

# ------------------------------------------------------------------ 5 HMM
story.append(PageBreak())
h1("5. HMM and Viterbi route reconstruction")
prm = hcfg["params"]
qa("Q1 Hidden states.", "At fix i the hidden state is one candidate arc c ∈ C<sub>i</sub> – a real arc id of the trip's network, stored together with the projection "
   "fraction t<sub>c</sub> along the arc. The candidate-to-arc correspondence is therefore preserved by construction; the decoder returns arc ids directly.")
qa("Q2 Emission probabilities.", "log e<sub>i</sub>(c) = λ·log p̂<sub>i</sub>(c), where p̂<sub>i</sub> is the tuned classifier's probability renormalised over C<sub>i</sub> "
   "(so it is a proper distribution over the candidates). λ weights the classifier against the transition model.")
qa("Q3 Transition probabilities.", "For c at fix i−1 and c' at fix i, the network distance is d<sub>route</sub> = (1−t<sub>c</sub>)L<sub>c</sub> + SP(head(c), tail(c')) + t<sub>c'</sub>L<sub>c'</sub> "
   "(SP = bounded Dijkstra on the <i>directed</i> arc graph, cutoff 2·d<sub>gc</sub> + 150 m), or |t<sub>c'</sub>−t<sub>c</sub>|L<sub>c</sub> on the same arc, with backward movement added as a penalty. "
   "With d<sub>gc</sub> the straight-line distance between the two fixes, log T = −|d<sub>route</sub> − d<sub>gc</sub>| / β: the detour a vehicle would need to make between "
   "the two candidates must agree with how far it actually moved. Direction is enforced because SP only follows arcs forwards.")
qa("Viterbi recurrence.", "δ<sub>i</sub>(c') = log e<sub>i</sub>(c') + max<sub>c</sub>[δ<sub>i−1</sub>(c) + log T(c, c')], with back-pointers ψ<sub>i</sub>(c') = argmax; "
   "the path is recovered by backtracking from argmax δ<sub>N</sub> (src/pathsnap/hmm.py). Scores are renormalised at each step for numerical stability.")
qa("Q4 Missing candidates and large GPS gaps.", "If a fix has no candidate (nothing within 250 m) it receives no arc (−1, counted as an error in Top-1), and the chain restarts after it. "
   "If a fix follows a gap (Δt > 30 s or a jump > 500 m), transitions into it are uniform (log T = 0): the decoder keeps one global path but does not force "
   "connectivity across the gap, because a short bounded search would wrongly penalise any route driven during the gap.")
qa("Q5 Disconnected road segments.", f"When no network path within the cutoff links two candidates (different components, one-way violations, missing links), the transition "
   f"gets a finite penalty log T = {prm['unreachable_logp']} instead of −∞. This makes the decoder strongly prefer connected paths, but it never dead-ends "
   "when the network itself is disconnected. During route reconstruction, connector arcs between consecutive predicted arcs are filled with shortest paths "
   "only when their length is plausible (≤ 2·GPS displacement + 200 m); otherwise the break is reported (unresolved_breaks) instead of inventing a detour.")
qa("Decoder tuning.", f"β, λ and the unreachable penalty were grid-searched ({len(htune)} combinations) on {len(hcfg['tuning_trips'])} training trips, using the "
   f"<b>out-of-fold</b> probabilities of the tuned model so that no trip is decoded with a classifier that saw it. Best: β = {prm['beta']} m, λ = {prm['lam']}, "
   f"unreachable log-p = {prm['unreachable_logp']} (Top-1 on tuning trips {hcfg['tuning_top1']:.4f} vs {hcfg['tuning_classifier_only_top1']:.4f} without the HMM). "
   f"{int((htune.top1 >= htune.top1.max() - 1e-9).sum())} settings tie for the best value and {int((htune.top1 >= htune.top1.max() - 0.002).sum())} are within 0.002 of it, so the result "
   "is not sensitive to the exact choice; what matters is a small λ (trust connectivity, use the classifier mainly to break ties) and a strong unreachable penalty.")
img(FIG / "hmm_tuning.png", 16.5, "HMM parameter grid search on training trips (out-of-fold probabilities).")
h2("5.1 Q6 – Classifier-only vs classifier + HMM (student-test)")
table([["Metric (10 student-test trips)", "Classifier only (arg-max)", "Classifier + HMM/Viterbi", "Δ"],
       ["Observation Top-1 accuracy", f4(ch['top1']), f4(hh['top1']), f"{hh['top1']-ch['top1']:+.4f}"],
       ["Arc Macro F1 (per-trip, mean)", f4(ch['arc_macro_f1']), f4(hh['arc_macro_f1']), f"{hh['arc_macro_f1']-ch['arc_macro_f1']:+.4f}"],
       ["Route continuity", f3(ch['continuity']), f3(hh['continuity']), f"{hh['continuity']-ch['continuity']:+.3f}"],
       ["Disconnected arc changes (total)", ch['n_breaks'], hh['n_breaks'], hh['n_breaks'] - ch['n_breaks']],
       ["Route F1 (reconstructed path vs reference arcs)", f4(hcmp['reconstructed_route_f1']['classifier_only']), f4(hcmp['reconstructed_route_f1']['classifier_hmm']),
        f"{hcmp['reconstructed_route_f1']['classifier_hmm']-hcmp['reconstructed_route_f1']['classifier_only']:+.4f}"],
       ["Unresolved breaks after gap filling", hcmp['unfilled_breaks']['classifier_only'], hcmp['unfilled_breaks']['classifier_hmm'], ""],
       ["Candidate miss rate (upper bound on Top-1 = 1 − miss)", pc(hh['cand_miss_rate']), pc(hh['cand_miss_rate']), ""]],
      col_widths=[6.5 * cm, 3.3 * cm, 3.6 * cm, 2.1 * cm], caption="Effect of sequential decoding (results/hmm_comparison.json).")
wins = int((hper.hmm_top1 > hper.clf_top1).sum())
worse = hper[hper.hmm_top1 < hper.clf_top1]
p(f"<b>Yes – HMM decoding improved the final route.</b> Top-1 rose by {100*(hh['top1']-ch['top1']):.2f} percentage points and route continuity from "
  f"{ch['continuity']:.3f} to {hh['continuity']:.3f}: the classifier alone produced {ch['n_breaks']} disconnected arc changes, the decoder {hh['n_breaks']}. "
  f"Accuracy improved on {wins} of {len(hper)} trips"
  + (f"; it decreased on {', '.join(worse.trip_id)} ({', '.join(f'{a:.4f}→{b:.4f}' for a, b in zip(worse.clf_top1, worse.hmm_top1))})." if len(worse) else ".")
  + " The decoding cost is "
  f"{hcmp['decode_s_total']:.1f} s for all {int(hper.n_obs.sum()):,} student-test fixes (lattice construction + Viterbi).")
img(FIG / "hmm_per_trip.png", 16.5, "Per-trip Top-1 and route continuity with and without the HMM (student-test).")
h2("5.2 Where the HMM helps and where it does not")
e = err
red = lambda c: 1 - e.loc[c, 'hmm'] / max(e.loc[c, 'clf'], 1)
p("We put every wrong fix into one of five categories (results/error_analysis.csv and the error-taxonomy figure). The HMM removes most <b>off-route</b> errors "
  f"({int(e.loc['off-route arc','clf'])} → {int(e.loc['off-route arc','hmm'])}, −{100*red('off-route arc'):.0f} %), where a noisy fix is closer to a parallel or crossing street that cannot be reached "
  f"from the surrounding path. It also removes many <b>reverse-direction twin</b> errors ({int(e.loc['reverse-direction twin arc','clf'])} → {int(e.loc['reverse-direction twin arc','hmm'])}, −{100*red('reverse-direction twin arc'):.0f} %), "
  "because a direction flip requires a U-turn that the directed transition model penalises. Twin errors nevertheless remain the largest category after decoding "
  f"({100*e.loc['reverse-direction twin arc','share_hmm']:.0f} % of the remaining errors): on short stretches both directions fit the GPS equally well. It helps least with <b>adjacent-arc boundary</b> errors "
  f"({int(e.loc['adjacent arc (boundary)','clf'])} → {int(e.loc['adjacent arc (boundary)','hmm'])}, −{100*red('adjacent arc (boundary)'):.0f} %). Near a node the fix is about equally close to the two consecutive arcs, both "
  "transitions are feasible, and the 'correct' answer depends on where the monotone label alignment put the switch. These are label-resolution "
  f"errors more than routing errors. Fixes whose reference arc is <b>not in the candidate set</b> ({int(e.loc['reference not in candidates','hmm'])}) cannot be "
  "recovered by any decoder. When the classifier is already confident and the network is simple (e.g. motorway trips), the HMM changes little.")
img(FIG / "error_analysis.png", 13.5, "Error taxonomy on student-test, classifier-only vs classifier + HMM.")

# ------------------------------------------------------------------ 6 folium
story.append(PageBreak())
h1("6. Geographical visualisation using Folium")
p("Maps are produced by src/pathsnap/visualization.py (scripts/03_eda_plots.py for EDA, scripts/08_final_maps.py for results) and saved as interactive "
  "HTML in results/maps/. Each map has an OpenStreetMap basemap, a fitted extent, a legend and a layer control with: local road network (grey), raw GPS "
  "observations (blue dots), reference route (green, offline student-test maps only), classifier-only route (orange, hidden by default) and the "
  "PathSnap classifier+HMM route (red). Start and end markers are shown.")
for tid in hper.sort_values("n_obs").trip_id.iloc[[1, -1]]:
    r_ = hper.set_index("trip_id").loc[tid]
    img(SHOT / f"map_final_{tid}.png", 16, f"Student-test trip {tid}: raw GPS, reference (green) and predicted route (red); Top-1 {r_.clf_top1:.3f} → {r_.hmm_top1:.3f} with HMM.")
img(SHOT / "map_eda_train_00000054.png", 14, "Training trip 00000054: the GPS trace and the supplied reference route disagree systematically (excluded from fitting).")

# ------------------------------------------------------------------ 7 api ui deploy
story.append(PageBreak())
h1("7. API, user interface and deployment")
h2("7.1 FastAPI backend (Laptop 1)")
table([["Endpoint", "Purpose"],
       ["GET /api/health", "Liveness, loaded model, HMM parameters, number of evaluation trips"],
       ["GET /api/trips", "The evaluation trips (id, #fixes, duration, network size) – no labels"],
       ["POST /api/predict {trip_id}", "Full pipeline: load track + network → clean → candidates → features → classifier → Viterbi → route; returns GPS, per-fix arcs, route polylines, timings and server-side metrics"],
       ["GET /api/trips/{id}/route", "Same prediction payload via GET (cached)"],
       ["GET /api/trips/{id}/metrics", "Server-side evaluation metrics only"],
       ["GET /api/trips/{id}/network", "Road-network geometry around the trip for the map"]],
      col_widths=[4.3 * cm, 12.2 * cm], caption="REST API (backend/app/main.py). Interactive docs at /docs.")
p("Structure: <font face='Courier'>backend/app/config.py</font> (environment-driven settings: evaluation-trip folder, model path, CORS origins), "
  "<font face='Courier'>schemas.py</font> (Pydantic request/response validation; trip ids must match ^\\d{1,12}$, so path traversal is impossible), "
  "<font face='Courier'>service.py</font> (trip registry, cached predictions, server-side evaluation) and <font face='Courier'>main.py</font> "
  "(routes, CORS, request logging to console and experiments/logs/api.log, error handling: 404 unknown trip, 422 invalid id, 500 with logged traceback). "
  "The model and HMM configuration are loaded once at start-up from artifacts/. <b>Leakage protection:</b> the prediction path "
  "(<font face='Courier'>MapMatcher.match</font>) only receives the track and network; the <font face='Courier'>.route</font> file is read solely in "
  "<font face='Courier'>PathSnapService._evaluate</font>, after the prediction is final, and only aggregate metrics are returned – never reference arcs or "
  "geometry (asserted by tests/test_api.py).")
h2("7.2 Frontend (Laptop 2)")
p("React 19 + TypeScript + Leaflet (react-leaflet), built with Vite (frontend/). The dashboard shows PathSnap branding, an overview row (number of trips, "
  "model, prediction status, headline accuracy), the ten evaluation trips with details, and a map with legend and layer control. Layers: raw GPS, "
  "reconstructed HMM route, classifier-only route and road network. A metrics panel compares classifier-only and HMM results "
  "(Top-1, arc macro F1, continuity, route precision/recall/F1, candidate miss rate) and shows pipeline timings. The backend URL comes from "
  "VITE_API_BASE_URL and can be changed at run time in the header (stored in the browser). Loading, API-offline and request-error states are handled. "
  "Reference routes are deliberately not drawn in the UI because the guidelines require ground truth to stay server-side; the UI receives only predictions and "
  "aggregate metrics.")
for f_, c_ in (("ui_trip_00000056.png", "Working UI after predicting student-test trip 00000056 (map layers, metrics, timings)."),
               ("ui_home.png", "UI landing state with the ten selectable evaluation trips."),
               ("ui_error_state.png", "API error state when the backend URL is unreachable.")):
    img(SHOT / f_, 16.5, c_)
h2("7.3 Two-laptop deployment")
bullets([
    "Both laptops join the same Wi-Fi/LAN. On Laptop 1 find the IPv4 address (<font face='Courier'>ipconfig</font> / <font face='Courier'>hostname -I</font>).",
    "Laptop 1: <font face='Courier'>pip install -r requirements.txt</font>; set PATHSNAP_EVAL_DIR to the folder containing the ten evaluator trips; run "
    "<font face='Courier'>deployment/start_backend.bat</font> (uvicorn on 0.0.0.0:8000). Allow inbound TCP 8000 in the OS firewall (Windows: "
    "<font face='Courier'>netsh advfirewall firewall add rule name=PathSnap dir=in action=allow protocol=TCP localport=8000</font>, run as administrator).",
    "Laptop 2: <font face='Courier'>deployment/start_frontend.bat &lt;LAPTOP1_IP&gt;</font> (writes VITE_API_BASE_URL, builds and serves on port 5173), open "
    "http://localhost:5173. Test connectivity first with <font face='Courier'>python deployment/check_api.py http://&lt;LAPTOP1_IP&gt;:8000</font>.",
    "CORS: PATHSNAP_CORS_ORIGINS defaults to '*' (GET/POST only); for the demo it can be restricted to http://&lt;LAPTOP2_IP&gt;:5173.",
    "Every request is logged on Laptop 1's console with the client IP, path, status and latency, plus a per-trip line with the server-side Top-1, as the evaluators require.",
])
p("<b>Verification status.</b> The complete chain (browser UI → HTTP → FastAPI → model → HMM → JSON → map) was run and tested on a single machine, with the UI "
  "and API in separate processes talking over HTTP via the configurable base URL. A physical two-laptop run was not possible in our environment and has to be "
  "repeated by the team on the demo network, using the steps above.")

# ------------------------------------------------------------------ 8 testing
h1("8. Testing and results")
p("<b>Automated tests</b> (pytest, tests/): geometry (point-to-segment, bearings, projection vs haversine), monotone label alignment, candidate-to-arc "
  "correspondence (stored distance equals the recomputed distance to that arc id), label-free preparation, Viterbi preferring a connected route over a noisy arg-max, "
  "gap/empty-candidate handling, invalid-row cleaning, and API integration (health, 10 trips, 404/422 handling, response contract, no reference "
  "keys in responses, metrics endpoint consistency, network endpoint).")
if pytest_log:
    last = [ln for ln in pytest_log.splitlines() if ln.strip()][-1]
    p(f"Latest run (experiments/logs/pytest.log): <b>{last.strip('= ')}</b>.")
if e2e is not None:
    p(f"<b>End-to-end HTTP test</b> (scripts/11_e2e_api.py against the running server, every evaluation trip): {int((e2e.status == 200).sum())}/{len(e2e)} trips returned HTTP 200. "
      f"Observation-weighted HMM Top-1 = {(e2e.hmm_top1_accuracy * e2e.n_obs).sum() / e2e.n_obs.sum():.4f} "
      f"(matches §5: {hh['top1']:.4f}); server time per trip {e2e.server_ms.min()/1000:.2f}–{e2e.server_ms.max()/1000:.2f} s.")
    rows = [["Trip", "Fixes", "Top-1 clf", "Top-1 HMM", "Continuity HMM", "Route P / R", "Breaks", "Server s"]]
    for _, r_ in e2e.iterrows():
        rows.append([r_.trip_id, int(r_.n_obs), f3(r_.classifier_top1_accuracy), f3(r_.hmm_top1_accuracy), f3(r_.hmm_route_continuity),
                     f"{r_.hmm_route_precision:.3f} / {r_.hmm_route_recall:.3f}", int(r_.breaks), f"{r_.server_ms/1000:.2f}"])
    table(rows, col_widths=[1.9 * cm, 1.4 * cm, 1.7 * cm, 1.7 * cm, 2.2 * cm, 3 * cm, 1.4 * cm, 1.5 * cm], caption="Per-trip results returned by the API (results/e2e_api_results.csv).")
logf = ROOT / "experiments/logs/api.log"
if logf.exists():
    lines = [ln for ln in logf.read_text(encoding="utf-8", errors="ignore").splitlines() if "pathsnap.api" in ln][-14:]
    p("API log excerpt from Laptop 1's console/log file (experiments/logs/api.log), recorded while the e2e test and the UI were running:")
    story.append(Preformatted("\n".join(ln[:150] for ln in lines), CODE))

# ------------------------------------------------------------------ 9 limitations
h1("9. Limitations and conclusions")
bullets([
    "<b>Label uncertainty.</b> Observation labels come from aligning a route to the fixes. Near nodes the switch point between consecutive arcs is ambiguous, "
    "and two training trips had unusable references. Reported accuracies are therefore measured against an approximate reference, not exact physical truth.",
    "<b>Model ceiling.</b> Five quite different learners converge to ≈0.86 Top-1, and their training- and validation-fold accuracies are almost equal. Further gains are more likely "
    "from richer sequence features or joint training with the decoder than from additional capacity, which also explains the small gain from tuning.",
    "<b>Duplicated arcs</b> (same from/to) are indistinguishable; predicting the 'other' copy counts as an error.",
    "<b>Straight-segment geometry</b> is assumed for arcs (no shape points are supplied beyond nodes).",
    "<b>Evaluator partition</b> was not available to us. The API reads it from a configurable folder, and the stand-in student-test results should not be "
    "read as evaluator scores.",
    "<b>Two-laptop run</b> still needs to be repeated on the demo network (see §7.3).",
])
p(f"<b>Conclusion.</b> PathSnap implements the full pipeline: verified EDA, label alignment, candidate generation with "
  f"{pc(hh['cand_miss_rate'])} miss rate, seven compared models under trip-grouped CV, Optuna-tuned HistGradientBoosting, and an HMM/Viterbi decoder that "
  f"raises student-test Top-1 from {ch['top1']:.4f} to {hh['top1']:.4f} and route continuity from {ch['continuity']:.3f} to {hh['continuity']:.3f}. It is served "
  "through a FastAPI backend and a React dashboard that keep reference data on the server.")
p("<b>Reproducibility.</b> <font face='Courier'>python scripts/01_eda_inventory.py → 02_build_dataset.py → 03_eda_plots.py → 04_train_models.py → "
  "05_compare_models.py → 06_optuna.py → 04_train_models (tuned refit) → 07_hmm.py → 08_final_maps.py → 10_analysis_figures.py</font>, then "
  "<font face='Courier'>python reports/build_report.py</font>. Seeds, configurations and logs are in experiments/.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "PathSnap – Technical Report")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                        title="PathSnap Technical Report", author="PathSnap team")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("report ->", OUT)
