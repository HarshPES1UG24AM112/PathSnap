# PathSnap – Project Status

_Last updated: 2026-11-18. Technical report: `reports/PathSnap_Technical_Report.pdf`. Progress report and viva guide: `reports/PROGRESS_REPORT_AND_VIVA_GUIDE.pdf`._

## Completed (executed; outputs in repo)

| Area | What was done | Evidence |
|---|---|---|
| Dataset | 80 train + 10 student_test trips. The evaluator partition was not supplied. | — |
| EDA | Inventory, data-quality figures, Folium maps | `results/eda/`, `results/figures/eda/`, `results/maps/eda_*.html` |
| Preprocessing | Cleaning, local projection, monotone route→GPS label alignment, candidates within 50 m, 23 features | `src/pathsnap/`, `scripts/02_build_dataset.py` |
| Models | 7 models with 5-fold GroupKFold plus student-test evaluation | `experiments/models/`, `artifacts/models/`, `results/model_comparison.csv` |
| Bayesian optimisation | Optuna TPE on HistGB, 40 trials. CV Top-1 0.8615 → 0.8651; student-test 0.8670 → 0.8744 | `experiments/optuna/`, `results/tuned_vs_untuned.csv` |
| HMM decoder tuning | Tuned on training trips with out-of-fold scores (192 configs). Final: β = 40, λ = 0.05, unreachable log-p = −25 | `artifacts/hmm_config.json`, `results/hmm_tuning.csv` |
| HMM on student-test | Top-1 0.8744 → **0.9378**; continuity 0.791 → 0.913; route F1 0.860 → 0.972; all 10 trips improved | `results/hmm_comparison.json`, `results/hmm_student_test_per_trip.csv`, `results/error_analysis.csv` |
| Final maps | Final Folium maps for all 10 student-test trips, with PNG screenshots | `results/maps/final_*.html`, `reports/screenshots/` |
| Backend | FastAPI served the final model over HTTP. All 10 trips returned 200; obs-weighted Top-1 0.9378, matching the offline result | `results/e2e_api_results.csv`, `experiments/logs/api.log` |
| Frontend | React UI tested in a browser against the live API; no console errors | `reports/screenshots/ui_*.png` |
| Tests | **20/20 pass**: 7 unit, 13 API, including all 10 trips and the no-leakage check | `experiments/logs/pytest.log` |
| Technical report | 20-page PDF; every number is read from result files | `reports/PathSnap_Technical_Report.pdf` |

## Pending (team actions)
1. Fill in the team names and SRNs on the report's title page (`reports/build_report.py`, "Team" row), then re-run `python reports/build_report.py`.
2. Two-laptop dry run on the demo network (`deployment/`, README "Running the system").
   - Take a screenshot of the Laptop 1 console showing API requests; the guidelines require it.
3. On evaluation day, set `PATHSNAP_EVAL_DIR` to the evaluator-trip folder on Laptop 1.

## Known issues / decisions
- Train trips 00000033 and 00000054 are excluded from fitting and CV. Their median GPS→reference distance is over 120 m, so the reference is unreliable.
- The tuned model shows a train/validation gap (0.928 vs 0.865 Top-1). Validation and student-test scores still improved.
- The largest remaining error type is reverse-direction twin arcs (52 % of errors after the HMM).
- `data/raw/` (dataset) and `data/processed/` (regenerable features) are git-ignored. See `data/README.md`.
