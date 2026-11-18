# PathSnap – Project Status

_Last updated: 2026-11-18. The full progress report and viva guide is in `reports/PROGRESS_REPORT_AND_VIVA_GUIDE.pdf` (source: `.md`)._

## Completed (executed, outputs in repo)
- **EDA:** `scripts/01_eda_inventory.py`, `scripts/03_eda_plots.py` → `results/eda/`, `results/figures/eda/`, `results/maps/eda_*.html`.
  - Dataset: 80 train + 10 student_test trips. The evaluator partition was not supplied.
- **Preprocessing, labels, candidates, features:** `src/pathsnap/` and `scripts/02_build_dataset.py`.
  - Monotone route alignment produces one label per GPS point.
  - Candidates: arcs within 50 m, at most 12 per point.
  - 23 features.
- **Seven models with 5-fold GroupKFold:** `scripts/04_train_models.py` → `experiments/models/*.json`, `artifacts/models/*.joblib`.
  - Comparison: `results/model_comparison.csv`, `results/figures/models/`.
- **Optuna (TPE) on HistGradientBoosting:** 40 trials → `experiments/optuna/`.
  - CV Top-1: 0.8615 → 0.8651. Student-test Top-1: 0.8670 → 0.8744.
  - Tuned model: `artifacts/models/hist_gbm_tuned.joblib`.
- **HMM/Viterbi:** `src/pathsnap/hmm.py`, `scripts/07_hmm.py`.
  - Decoder tuned on training trips using out-of-fold scores (144-config grid): best β = 10, λ = 0.15, unreachable penalty −50 (`artifacts/hmm_config.json`). Tuning-trip Top-1: 0.868 → 0.944.
  - Student-test, measured with the previous setting (β = 5, λ = 0.5, −25): Top-1 0.874 → 0.936; continuity 0.791 → 0.912; improved on all 10 trips.
- **Unit tests:** `tests/test_core.py`, 7/7 pass.

## Built but not yet verified end-to-end
- **FastAPI backend (`backend/`):** smoke-tested in-process with the untuned model. Not yet run with the final artifacts.
- **React + TypeScript + Leaflet UI (`frontend/`):** `npm run build` succeeds. Not yet tested in a browser against the API.
- **Remaining scripts, written but not run:**
  - `tests/test_api.py`
  - `scripts/08_final_maps.py`
  - `scripts/09_screenshots.py`
  - `scripts/10_analysis_figures.py`
  - `scripts/11_e2e_api.py`
  - `reports/build_report.py` (technical report PDF)

## Next steps (in order)
1. Add λ ∈ {0.05, 0.1} to `GRID` in `scripts/07_hmm.py`, then run `python scripts/07_hmm.py hist_gbm_tuned`. This re-runs the student-test comparison with the final decoder setting.
2. `python scripts/08_final_maps.py`, then `python scripts/10_analysis_figures.py`.
3. Start the API (`deployment/start_backend.bat`) and the UI (`cd frontend && npm run dev`). Then run:
   - `python scripts/11_e2e_api.py http://localhost:8000`
   - `python scripts/09_screenshots.py maps`
   - `python scripts/09_screenshots.py ui http://localhost:5173 00000052 00000056`
4. `python -m pytest -q > experiments/logs/pytest.log`
5. `python reports/build_report.py`. Then fill in the README results table and the team names.
6. Do a two-laptop dry run on the demo network.

## Known issues / decisions
- Train trips 00000033 and 00000054 are excluded from fitting and CV. Their median GPS→reference distance is over 120 m, so the reference is unreliable.
- The tuned model shows a train/validation gap (0.928 vs 0.865 Top-1). Validation and student-test scores still improved.
- `data/raw/` (dataset) and `data/processed/` (regenerable features) are git-ignored. See `data/README.md`.
