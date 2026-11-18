# PathSnap — Progress Report & Viva Preparation Guide

_Status as of 2026-11-18 (final update after all experiments). Every number here comes from files in the repository; the source file is named next to it._

---

## Part A — Progress report

### A.1 Status at a glance

| # | Work package | Status | Evidence (files) |
|---|---|---|---|
| 1 | Read the 5 official documents | ✅ Done | — |
| 2 | Dataset inspection + EDA | ✅ Done | `scripts/01_eda_inventory.py`, `scripts/03_eda_plots.py`, `results/eda/*.csv`, `results/figures/eda/*.png`, `results/maps/eda_*.html` |
| 3 | Preprocessing, label alignment, candidate generation, 23 features | ✅ Done | `src/pathsnap/{preprocessing,labels,network,features,pipeline}.py`, `scripts/02_build_dataset.py` |
| 4 | 7 models, 5-fold GroupKFold CV, student-test evaluation | ✅ Done | `scripts/04_train_models.py`, `experiments/models/*.json`, `artifacts/models/*.joblib` |
| 5 | Model comparison table + figures | ✅ Done | `scripts/05_compare_models.py`, `results/model_comparison.csv`, `results/figures/models/` |
| 6 | Bayesian optimisation (Optuna, 40 trials) + tuned refit | ✅ Done | `scripts/06_optuna.py`, `experiments/optuna/{trials.csv,best_params.json,study.db}`, `experiments/models/hist_gbm_tuned.json` |
| 7 | HMM/Viterbi implementation | ✅ Done | `src/pathsnap/hmm.py`, 2 unit tests |
| 8 | HMM decoder tuning (on training trips) | ✅ Done (192 configs) | `results/hmm_tuning.csv`, `artifacts/hmm_config.json` |
| 9 | Classifier-only vs HMM on student-test | ✅ Done with the final decoder setting | `results/hmm_comparison.json`, `results/hmm_student_test_per_trip.csv` |
| 10 | FastAPI backend | ✅ Served the final model over HTTP; all 10 trips returned 200 | `backend/app/*.py` |
| 11 | React + TypeScript + Leaflet frontend | ✅ Tested in a browser against the live API; no console errors | `frontend/src/*` |
| 12 | Unit tests | ✅ 7/7 pass | `tests/test_core.py` |
| 13 | API integration tests | ✅ 13/13 pass (20/20 in total) | `tests/test_api.py` |
| 14 | Final Folium result maps | ✅ 10 maps plus screenshots | `scripts/08_final_maps.py` |
| 15 | Analysis figures (Optuna history, HMM grid, error taxonomy) | ✅ Done | `scripts/10_analysis_figures.py` |
| 16 | UI screenshots, end-to-end HTTP test on all 10 trips | ✅ Done (`results/e2e_api_results.csv`) | `scripts/09_screenshots.py`, `scripts/11_e2e_api.py` |
| 17 | Technical report PDF | ✅ 20 pages (`reports/PathSnap_Technical_Report.pdf`) | `reports/build_report.py` |
| 18 | README, `.gitignore`, deployment scripts | ✅ Done, including the README results table | `README.md`, `deployment/*` |
| 19 | Physical two-laptop test | ❌ Cannot be done from here; the team must run it on the demo network | `deployment/` |
| 20 | Git repo / GitHub push | ✅ Public: https://github.com/HarshPES1UG24AM112/PathSnap | — |

### A.2 Remaining steps (team)
1. Add the team names and SRNs to the report title page (`reports/build_report.py`), then run `python reports/build_report.py`.
2. Do a two-laptop dry run on the demo network and screenshot Laptop 1's console showing the API logs.
3. On demo day, point `PATHSNAP_EVAL_DIR` at the evaluator trips.

### A.3 Key results so far

**Dataset** (`results/eda/partition_summary.csv`)

| | train | student_test |
|---|---|---|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| NaN / invalid coordinates / non-increasing timestamps | 0 / 0 / 0 | 0 / 0 / 0 |
| Share of 1-second steps (trip mean) | 93.8 % | 98.2 % |
| Gaps > 5 s (largest) | 1,216 (153,064 s) | 44 (210 s) |
| Median nodes / arcs per trip network | 9,608 / 19,527 | 6,597 / 13,145 |
| Median GPS → reference-route distance | 4.5 m | 4.6 m |

- The evaluator-testing partition was **not included** in the zip files we received.
- Each trip comes with its **own road network**, and the trips are spread across many countries: longitudes range from −100° to +137°.
- Training trips **00000054** and **00000033** have a median GPS-to-route distance of about 138 m and 122 m (normal is about 4.5 m). Their reference routes are unreliable, so both trips are **excluded from fitting and CV**. This leaves 78 usable training trips.

**Candidate generation** (`results/eda/candidate_stats.csv`)
- Mean candidate-set size is about 7.9.
- Student-test candidate miss rate is **0.17 %**. This caps the best possible Top-1 at about 99.8 %.
- Nearest-arc baseline (always pick the closest arc): student-test Top-1 ≈ **0.54**.

**Seven models** (`results/model_comparison.csv`; CV = 5-fold GroupKFold over 78 training trips; ST = 10 student-test trips)

| Model | Family | CV Top-1 (± std) | CV Macro F1 | CV log loss | ST Top-1 | ST Macro F1 | ST log loss | ST continuity | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.8560 ± 0.0263 | 0.9181 | 0.1211 | 0.8618 | 0.9211 | 0.1099 | 0.762 | 1.5 |
| Gaussian Naive Bayes | Probabilistic generative | 0.5696 ± 0.0520 | 0.7505 | 1.3830 | 0.5444 | 0.7385 | 1.3080 | 0.440 | 2.6 |
| KNN (k=50) | Instance-based | 0.8482 ± 0.0267 | 0.9135 | 0.1414 | 0.8525 | 0.9157 | 0.1234 | 0.765 | 0.0 |
| Decision Tree | Tree | 0.8583 ± 0.0255 | 0.9194 | 0.1189 | 0.8619 | 0.9211 | 0.1024 | 0.763 | 14.9 |
| Random Forest | Bagging | 0.8619 ± 0.0263 | 0.9215 | 0.1038 | 0.8691 | 0.9253 | 0.0904 | 0.777 | 42.3 |
| HistGradientBoosting | Boosting | 0.8615 ± 0.0258 | 0.9213 | 0.1071 | 0.8670 | 0.9241 | 0.0915 | 0.773 | 4.6 |
| MLP (64-32) | Neural network | 0.8585 ± 0.0266 | 0.9195 | 0.1120 | 0.8744 | 0.9284 | 0.0923 | 0.792 | 59.7 |
| **HistGB tuned (Optuna)** | Boosting | **0.8651 ± 0.0262** | 0.9233 | 0.1058 | **0.8744** | 0.9284 | **0.0878** | 0.791 | 24.4 |

**Selected model: HistGradientBoosting.**
- Random Forest and HistGB are a statistical tie on CV Top-1. The paired per-fold differences (RF − HGB) are −0.0018, +0.0014, +0.0008, −0.0026 and +0.0042, with a mean of +0.0004.
- HistGB trains about 9× faster, which made a 40-trial Bayesian search affordable.
- The HistGB artifact is 0.2 MB, compared with 45 MB for Random Forest.
- HistGB optimises log loss directly, which gives good probabilities for the HMM emissions.

**Bayesian optimisation** (`experiments/optuna/best_params.json`)
- Setup: Optuna TPE sampler (seed 42), MedianPruner, 40 trials (20 completed, 20 pruned), 63 minutes.
- Objective: mean 5-fold GroupKFold Top-1.
- Trial 0 used the untuned defaults: 0.8620.
- Best: **0.8651**, with learning_rate 0.0705, max_leaf_nodes 129, min_samples_leaf 383, l2_regularization 0.143, max_features 0.971 and max_iter 550.
- Student-test Top-1: 0.8670 → 0.8744. Log loss: 0.0915 → 0.0878.
- Cost: refit time went from 4.6 s to 24.4 s. Inference went from 0.008 to 0.047 ms per fix.

**HMM / Viterbi**

Decoder tuning was done on 39 training trips using **out-of-fold** probabilities (`results/hmm_tuning.csv`, 192 settings):
- Classifier-only Top-1: **0.8684**.
- Best HMM: **0.9449**, with β = 40 m, λ = 0.05, unreachable log-p = −25 (`artifacts/hmm_config.json`).
- Several settings tie for the best value. What matters is a small λ (trust connectivity) and a strong unreachable penalty.

Student-test results with the final setting (`results/hmm_comparison.json`):

| Metric | Classifier only | Classifier + HMM |
|---|---|---|
| Observation Top-1 | 0.8744 | **0.9378** |
| Arc macro F1 (per-trip mean) | 0.7959 | **0.9154** |
| Route continuity | 0.791 | **0.913** |
| Disconnected arc changes | 1,141 | 322 |
| Reconstructed-route F1 | 0.8600 | **0.9717** |
| Unresolved breaks after gap filling | 318 | 3 |
| Decoding time (all 19,160 fixes) | — | 3.6 s |

The HMM improved Top-1 on **all 10** student-test trips (`results/hmm_student_test_per_trip.csv`). The hardest trip, 00000077, went from 0.730 to 0.846.

Errors on the student-test trips, classifier only → with HMM (`results/error_analysis.csv`):
- off-route: 782 → 244;
- reverse-direction twin arc: 1,076 → 614, still 52 % of the remaining errors;
- adjacent arc at a junction boundary: 390 → 254;
- reference arc not among the candidates: 41 → 41.

The API served all 10 trips over HTTP with the same overall Top-1 (0.9378). All 20 automated tests pass.

### A.4 Bugs found and fixed during development (worth mentioning honestly)
1. **HMM tuning misaligned the out-of-fold (OOF) scores.** They were attached before the two unreliable trips were removed, so the lengths didn't match, every score became NaN and tuning Top-1 showed 0.0. **Fix:** apply the exclusion first, then add an assertion that the lengths match. The tuning was then re-run.
2. **Route reconstruction invented detours.** It filled a 2.7 km connector path between two consecutive predicted arcs. **Fix:** a connector is now accepted only if its length is at most 2 × the GPS displacement + 200 m; otherwise the gap is reported as a break.

### A.5 Known limitations / open issues
- **Labels are approximate.** They come from aligning the route to the GPS points, so the exact switch point between two consecutive arcs near a node is ambiguous.
- **Duplicated arcs** (1,903 across all trips, with identical from/to nodes) cannot be told apart.
- **The tuned model overfits more than the untuned one.** Top-1 on training-fold trips is 0.928, against 0.865 on validation folds (untuned: 0.860 vs 0.862). Validation accuracy still improved, so the extra fit is useful but should be stated.
- **Not yet verified:** the physical two-laptop run. Everything else was tested on one machine over HTTP.

---

## Part B — How the system works (what you must be able to explain)

### B.1 The problem in one paragraph
GPS fixes are noisy: about 4.5 m typical error, with spikes and gaps. Map matching assigns each fix to the road arc the vehicle was actually on, and route reconstruction joins those arcs into a connected, drivable path. We do it in two stages:
1. A **classifier** scores every nearby arc for every fix.
2. An **HMM decoded with Viterbi** picks the most likely *sequence* of arcs, using road connectivity.

### B.2 The raw files (memorise this)

| File | Columns | Meaning |
|---|---|---|
| `.track` | lon, lat, t | One row per GPS fix, about 1 Hz |
| `.nodes` | lon, lat | Row index = node id |
| `.arcs` | from_node, to_node | Row index = arc id. Arcs are **directed**; a two-way street is two arcs (A→B and B→A) |
| `.route` | arc_id | The reference path as an **ordered list of arcs**, NOT one per GPS fix (trip 1 has 1,315 fixes but 525 route arcs) |

**Trap question:** "Does row i of `.track` correspond to row i of `.route`?" → **No.** We have to align the two ourselves (B.4).

### B.3 Pipeline (file → function)
1. **Load** (`io.py`). The route is loaded by a *separate* function, so the prediction code never touches it.
2. **Clean** (`preprocessing.py`):
   - drop NaN, out-of-range and (0,0) coordinates; sort by time; drop duplicate timestamps;
   - compute Δt, speed and heading. Heading uses a centred window that grows until the vehicle has moved at least 4 m, so jitter at standstill doesn't produce random headings;
   - flag spikes (`is_jump`: faster than 60 m/s to both neighbours) and gaps (`gap_before`: Δt > 30 s or a jump > 500 m).
3. **Project** (`geo.py`). Each trip gets its own local tangent plane: x = R·cos(φ0)·Δλ, y = R·Δφ.
   - **Why:** latitude/longitude are angles; one degree of longitude shrinks with latitude. Trips are under about 40 km across, so the error is < 0.1 %, far below GPS noise. A unit test checks it against haversine.
   - **Why not UTM:** trips sit in many different UTM zones, and our projection is cheaper.
4. **Candidates** (`network.py`, `features.py`):
   - each arc is sampled every 10 m and the samples go into a KD-tree, restricted to the track's bounding box + 300 m;
   - candidates are all arcs within **50 m** (keep the 12 nearest);
   - if none, take the 4 nearest within 250 m; if still none, the set is empty and the decoder skips that fix.
   - **Why 50 m:** the 95th percentile of GPS-to-reference distance is about 12–25 m, so 50 m captures almost everything (0.17 % miss on student-test) without too many candidates.
5. **Labels** (`labels.py`), training and evaluation only. A monotone dynamic programme assigns fix i to route position j(i), with j never decreasing as i increases, minimising the sum of min(distance, 100 m).
   - **Why monotone:** it follows the route order, so a road driven twice (a loop) is labelled with the right visit. Plain nearest-route-arc labelling would fail there.
6. **Features** (23 per candidate, see B.5).
7. **Classifier** gives P(candidate is the true arc), renormalised within each fix's candidate set.
8. **HMM/Viterbi** (`hmm.py`) gives one arc per fix.
9. **Route reconstruction** (`reconstruct_route`): collapse repeated arcs and fill missing connector arcs with shortest paths, but only plausible ones.
10. **Evaluation**: done server-side only, after the prediction is final.

### B.4 The key modelling decision: candidate ranking, not multi-class
- Arc ids are just row numbers in **each trip's own** network. Arc 2641 in trip 1 has nothing to do with arc 2641 in trip 2, so "arc id as class label" is meaningless.
- Instead, each training row is an **(observation, candidate arc) pair**, with the binary target "is this the reference arc?".
- Every row keeps `trip_id, obs, arc_id`, so a probability always maps back to a real road segment.
- Prediction is the per-observation **arg-max**.
- Positives are about 13 % of rows.

### B.5 Features and why each exists

| Group | Features | Reason |
|---|---|---|
| Geometry | dist, dist_rank, dist_minus_min, gauss_share, is_nearest, n_cands | How close the candidate is, absolutely and relative to the other candidates |
| Direction | heading_diff (0–180°), axis_diff (0–90°), heading_valid | Separates the two directions of a two-way street (the same distance!) and crossing streets |
| Kinematics | speed, dt, is_jump | Heading is unreliable when slow; spikes are untrustworthy |
| Arc / topology | arc_len, proj_t, at_endpoint, has_twin, out_deg_end, in_deg_start | Projection clamped to an end → probably the next arc; junctions are ambiguous |
| Temporal context | dist_prev, dist_next, ctx_mean_dist (±2 fixes) | The true road stays close over several fixes; noise doesn't |
| Connectivity | adj_prev_dist, adj_next_dist | Is a connected arc also a candidate for the previous/next fix? Label-free sequence evidence |

All features are deterministic: nothing is fitted, so nothing can leak. Scalers live inside the model pipelines and are fitted on training folds only.

### B.6 Validation protocol
- **GroupKFold, 5 folds, groups = trip_id.** Whole trips stay together.
  - **Why:** fixes from the same trip are highly correlated. A random row split would put neighbouring fixes in train and validation and inflate the scores (leakage).
- **Training rows:** only fixes whose reference arc is among the candidates. Others contain no positive and are pure noise.
- **Validation:** *all* fixes. A missed reference counts as wrong.
- **Final model:** refit on all 78 trips and evaluate once on the 10 student-test trips.
- **Seed:** 42 everywhere.

### B.7 Metrics (be able to define each)
- **Observation Top-1 accuracy (primary):** the fraction of fixes whose predicted arc equals the reference arc. It is exactly the task objective.
- **Precision / recall / Macro F1:** computed at candidate level with the **arg-max decision rule** (exactly one predicted positive per fix), so they need no threshold. Macro F1 averages the F1 of class 0 and class 1.
- **Log loss:** binary cross-entropy of the raw probabilities. It measures calibration, which matters because the probabilities become HMM emissions.
- **Arc macro F1:** per trip, macro F1 over arc ids (reference arc vs predicted arc), averaged over trips. Arc ids are only compared within a trip.
- **Route continuity:** of the places where the predicted arc changes, the fraction where the new arc starts at the node where the old one ends.
- **Candidate miss rate:** the fraction of fixes whose reference arc is not in the candidate set. No model can fix these.
- **Route precision / recall / F1:** set overlap between the reconstructed route arcs and the `.route` arcs.

### B.8 Why each model behaved as it did (the viva will ask)
- **Logistic Regression (0.856):** a linear, additive score. It can't express interactions (for example, "heading only matters when moving"), so it is slightly below the trees. Fast and well calibrated. Train ≈ validation, so no overfitting.
- **Gaussian NB (0.570, log loss 1.38):** **underfits and is mis-specified.** It assumes features are independent given the class, but dist, dist_rank, gauss_share, ctx_mean_dist and dist_prev/next all measure "closeness", so the same evidence is counted about 6 times. Its posteriors are extremely over-confident (high log loss), and it is barely better than picking the nearest arc (0.54).
- **KNN (0.848):** a lazy learner (k=50, distance-weighted, standardised features). It was fitted on a 60k-row sample because it stores all training points and compares each query against them. It is the slowest at inference (3.8 s for the student-test trips) and slightly less accurate.
- **Decision Tree (0.858):** captures thresholds and interactions. Depth 12 and min_samples_leaf 50 limit its variance. Probabilities are coarse (piece-wise constant).
- **Random Forest (0.862):** averaging 150 bagged trees lowers the variance of a single tree. Joint best on CV, but slow and 45 MB.
- **HistGB (0.862 → 0.865 tuned):** boosting reduces bias by fitting the residual errors; histogram binning makes it fast. Early stopping ended the untuned fit at 125 of 300 iterations.
- **MLP (0.859 CV, 0.874 ST):** the best untuned student-test score, but we select on **CV over 78 trips**, not on a 10-trip sample. Some folds hit the 40-epoch cap (ConvergenceWarning).

**Big-picture insight:** five very different models all plateau around 0.86 Top-1, and their train accuracy ≈ validation accuracy. That points to a **data/label ceiling**, not a model-capacity problem:
- reverse-direction twin arcs at the same distance;
- boundary ambiguity at nodes;
- the 0.17 % misses.

That is exactly why the **HMM gives the big jump** (+6.3 points) while tuning gives a small one (+0.3 points CV).

### B.9 HMM details (most likely viva topic)
- **Hidden state at fix i:** one of its candidate arcs (a real arc id, together with the position t along it).
- **Emission:** log e = λ·log p̂, where p̂ is the classifier probability renormalised over the fix's candidates. A small λ (0.05) means the decoder trusts connectivity strongly and the classifier mostly breaks ties.
- **Transition between consecutive fixes:**
  - **Different arcs:** d_route = (remaining length of the old arc) + (shortest directed path between them) + (distance into the new arc).
  - **Same arc:** d_route = |Δt|·L, with backward movement added as a penalty.
  - Then log T = −|d_route − d_gc| / β, where d_gc is the straight-line distance between the two fixes.
  - **Intuition:** the distance you would have to drive between the two candidates should match how far you actually moved. β (40 m) sets how forgiving this is.
- **Shortest path:** bounded Dijkstra on the **directed** graph (cutoff 2·d_gc + 150 m), cached per source node. Because only forward arcs are followed, one-way streets and direction are respected.
- **Unreachable pairs** (disconnected components, one-way violations, beyond the cutoff): log T = −25, a finite penalty rather than −∞, so Viterbi never dead-ends when the network itself is broken.
- **Gaps** (Δt > 30 s or a jump > 500 m): uniform transition (log T = 0), which effectively restarts the chain. A bounded search can't know what was driven during the gap.
- **Empty candidate set:** the fix gets no arc (−1, counted as wrong) and the chain restarts after it.
- **Viterbi recurrence:** δ_i(c′) = log e_i(c′) + max_c [δ_{i−1}(c) + log T(c, c′)]. Store back-pointers, then backtrack from the best final state. The scores are renormalised each step for numerical stability. Complexity is O(N·K²) with K ≤ 12.
- **Tuning:** a grid over β, λ and the unreachable penalty, run on 39 **training** trips using **out-of-fold** probabilities (each trip is scored by a model that never saw it). Student-test and evaluator data are not used.
- **Where it helps:** off-route errors (a noisy fix nearer a parallel street) and twin-direction errors (a direction flip needs a U-turn, which the directed graph penalises).
- **Where it doesn't help:**
  - boundary errors at nodes, where both arcs are reachable and the label itself is ambiguous;
  - fixes whose reference arc is missing from the candidates.

### B.10 Backend, frontend, deployment
- **Endpoints:** `GET /api/health`, `GET /api/trips`, `POST /api/predict {trip_id}`, `GET /api/trips/{id}/route`, `/metrics`, `/network`. Interactive docs are at `/docs`.
- **Validation:** `trip_id` must match `^\d{1,12}$` and be one of the evaluation folders, so path traversal is impossible. Unknown id → 404; malformed id → 422 (Pydantic).
- **Logging:** every request is logged to the console and `experiments/logs/api.log` with the client IP, path, status and latency, plus a per-trip line with Top-1. Evaluators want to see these logs on Laptop 1.
- **Leakage guard:**
  - `MapMatcher.match(track, network)` never sees the route;
  - `PathSnapService._evaluate` reads `.route` only **after** prediction and returns only aggregate metrics;
  - the UI never receives reference geometry, as the guidelines require ("ground truth must never be exposed to … UI");
  - a test checks that responses contain no reference keys.
- **Configuration through environment variables:** `PATHSNAP_EVAL_DIR` (point it at the evaluator trips on demo day), `PATHSNAP_CORS_ORIGINS`, `PATHSNAP_MODEL`, `PATHSNAP_HMM_CONFIG`.
- **Frontend:** React 19 + TypeScript + react-leaflet, built with Vite.
  - Panels: trip picker, map (GPS = blue, HMM route = red, classifier-only route = orange, road network = grey) with layer control and legend, and a metrics panel comparing classifier vs HMM.
  - The API URL can be changed at run time in the header.
  - Loading and error states are shown.
- **Two laptops:**
  1. Put both on the same LAN.
  2. Laptop 1: run `deployment\start_backend.bat` (uvicorn on `0.0.0.0:8000`) and open port 8000 in the firewall.
  3. Laptop 2: run `deployment\start_frontend.bat <LAPTOP1_IP>`.
  4. Check the connection first with `python deployment/check_api.py http://<IP>:8000`.

---

## Part C — Likely viva questions with short answers

**Data**
1. *How many trips and fixes?* 80 train (208,417 fixes) and 10 student-test (19,160). The evaluator set was not given to us.
2. *Missing values?* None found. We still handle NaN, invalid and duplicate rows for unseen data. The real problems were gaps (up to 153,064 s), speed spikes (1,531 steps > 50 m/s), fragmented networks, duplicate arcs and two unreliable references.
3. *Why not Euclidean distance on lat/lon?* One degree of longitude ≈ 111 km·cos(latitude), so degrees aren't metres. We use a local metric projection per trip.
4. *How did you get a label per GPS fix?* With the monotone DP alignment of the route to the fixes (B.3, step 5).
5. *Why exclude trips 33 and 54?* Their GPS is about 120–140 m from the reference route, so the labels would be wrong. The decision was made from training data only, with a documented rule (median > 50 m).

**Modelling**
6. *Why not use arc id as the class?* Arc ids are local to each trip's network → we use candidate ranking (B.4).
7. *How did you handle class imbalance?* The arg-max-per-fix formulation makes the threshold irrelevant. We did not reweight, because that would distort the probabilities the HMM needs.
8. *Primary metric and why?* Observation Top-1: it is the task objective, threshold-free, and the quantity the HMM improves.
9. *Why GroupKFold?* To avoid leakage between correlated fixes of the same trip.
10. *Why HistGB over Random Forest?* They tie on CV (mean paired difference +0.0004); HistGB is 9× faster to train (so Optuna was feasible), 200× smaller, and optimises log loss.
11. *Why not choose the MLP, which had the best student-test score?* Ten trips is a small sample; CV over 78 trips is more reliable, and the MLP is lower there (0.8585).
12. *Overfitting evidence?* Train-fold vs validation Top-1 is almost equal for all untuned models. The tuned HistGB shows a gap (0.928 vs 0.865), but its validation and student-test scores still improved.
13. *Why is Naive Bayes so bad?* Correlated features violate the independence assumption, so the evidence is double-counted and the model is over-confident (log loss 1.38).

**Optimisation**
14. *Why Bayesian rather than grid search?* TPE models which regions of the search space score well and samples there, so it needs far fewer trials in a 6-dimensional space. The pruner stops bad trials early.
15. *Search space?* learning_rate log [0.02, 0.3]; max_leaf_nodes log [15, 255]; min_samples_leaf log [10, 1000]; l2_regularization log [1e-4, 10]; max_features [0.3, 1]; max_iter [100, 800].
16. *Objective?* Mean 5-fold GroupKFold Top-1 on the training trips.
17. *Trials?* 40 (20 completed, 20 pruned), 63 minutes.
18. *Did it help?* CV +0.0031 and student-test +0.0074 Top-1; log loss improved. A small gain, consistent with the ceiling argument.

**HMM**
19. *States, emissions, transitions?* See B.9.
20. *What if two consecutive fixes suggest disconnected roads?* The transition gets a −25 log penalty (not impossible). Route reconstruction fills only plausible connector paths; otherwise it reports a break.
21. *Did the HMM help?* Yes: student-test Top-1 0.874 → 0.938, continuity 0.79 → 0.91, reconstructed-route F1 0.86 → 0.97, and it improved all 10 trips.
22. *Where did it not help?* Boundary errors at nodes and missing candidates.
23. *How were β and λ tuned without leakage?* A grid on training trips with out-of-fold probabilities.

**System and integrity**
24. *How do you guarantee the ground truth isn't leaked?* B.10 leakage guard, plus a test.
25. *What happens on demo day?* Copy the 10 evaluator trip folders to Laptop 1, set `PATHSNAP_EVAL_DIR`, start the API, start the UI on Laptop 2 pointing to Laptop 1's IP, select a trip, and show the API logs.
26. *Did you use AI?* Yes, an AI coding assistant. All numbers come from code that was actually run, and this is disclosed in the report and README.

---

## Part D — Things every team member should do before the viva
1. Run the pipeline once yourselves (README "Reproducing the experiments").
2. Open `results/maps/eda_train_00000001.html` and `eda_train_00000054.html` and explain what you see.
3. Read these files line by line — they are the heart of the project: `src/pathsnap/features.py` (`build_features`), `src/pathsnap/hmm.py` (`build_lattice`, `viterbi`) and `src/pathsnap/labels.py`.
4. Be able to draw the two-stage pipeline and the Viterbi trellis on a whiteboard.
5. Know the headline numbers by heart:
   - nearest-arc baseline 0.54;
   - best classifier 0.874 ST / 0.865 CV;
   - with HMM 0.938;
   - candidate miss rate 0.17 %.
