# PathSnap – Distributed Map Matching & Route Reconstruction

PathSnap matches noisy GPS observations to a road network and reconstructs the most likely route that the vehicle drove.
A supervised **candidate scorer** (Optuna-tuned HistGradientBoosting, chosen from 7 models) estimates how likely each nearby
road arc is to be the true one. An **HMM decoded with Viterbi** then combines these scores with directed road-network connectivity.
The system is served by a **FastAPI** backend (Laptop 1) and used through a **React + TypeScript + Leaflet** dashboard (Laptop 2).

> The full methodology, results and figures are in **[`reports/PathSnap_Technical_Report.pdf`](reports/PathSnap_Technical_Report.pdf)**.
> Progress, test status and known issues are tracked in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Results (student-testing trips, 10 trips; produced by the scripts below)

<!-- RESULTS_TABLE -->

## Architecture

```
            Laptop 2 (UI)                                   Laptop 1 (API + model)
 ┌──────────────────────────────┐   HTTP/JSON   ┌───────────────────────────────────────────────┐
 │ React + TS + Leaflet (Vite)  │ ────────────▶ │ FastAPI  /api/health /api/trips /api/predict │
 │ trip picker · map · metrics  │ ◀──────────── │   └─ MapMatcher (src/pathsnap/inference.py)    │
 └──────────────────────────────┘               │       clean → candidates → 23 features →       │
                                                │       HistGB probs → HMM/Viterbi → route        │
                                                │   └─ server-side evaluation (.route never sent) │
                                                └───────────────────────────────────────────────┘
```

**Formulation.** Arc ids are local to each trip's network, so they are never used as classes. Every row is an
*(observation, candidate arc)* pair carrying `trip_id, obs, arc_id`. The binary target is "candidate = reference arc", and
per-observation scores are renormalised into a distribution over candidate arcs, which the HMM uses as emissions.

## Repository layout

| Path | Contents |
|---|---|
| `src/pathsnap/` | Library: `io` (raw files), `geo` (local projection, distances), `preprocessing` (cleaning, kinematics), `network` (projected graph, KD-tree, bounded Dijkstra), `labels` (monotone route→fix alignment), `features` (candidates + 23 features), `models` (7-model zoo), `metrics`, `evaluation`, `hmm` (lattice, Viterbi, route reconstruction), `inference` (end-to-end `MapMatcher`), `visualization` (Folium) |
| `scripts/` | Numbered, reproducible experiment pipeline (EDA → dataset → models → comparison → Optuna → HMM → maps → figures → e2e) |
| `backend/app/` | FastAPI app (`main.py`), settings (`config.py`), Pydantic schemas, service layer with server-side evaluation |
| `frontend/` | React 19 + TypeScript + react-leaflet dashboard (Vite) |
| `artifacts/` | Final model `models/hist_gbm_tuned.joblib`, all compared models, `hmm_config.json` |
| `experiments/` | Per-model JSON (configs, folds, timings), Optuna study (`trials.csv`, `study.db`, `best_params.json`), logs |
| `results/` | CSV summaries, figures (`figures/`), interactive Folium maps (`maps/*.html`) |
| `reports/` | Technical report PDF, its generator `build_report.py`, screenshots |
| `deployment/` | Start scripts for both laptops, env example, connectivity check |
| `tests/` | Unit tests (`test_core.py`) and API integration tests (`test_api.py`) |
| `data/` | `raw/` (dataset, git-ignored) and `processed/` (generated features, git-ignored) |

## Installation

Requirements: Python ≥ 3.11 (tested with 3.13.7), Node.js ≥ 18 (tested with 22.19).

```bash
pip install -r requirements.txt
```

```bash
cd frontend && npm install
```

Put the dataset at `data/raw/train/<trip>/…` and `data/raw/student_test/<trip>/…` (the layout of the supplied `Student folder.zip`).

## Reproducing the experiments

Run each step from the repository root. Approximate run times on a 24-thread laptop are given in brackets.

```bash
python scripts/01_eda_inventory.py          # inventory + quality stats            (~1 min)
python scripts/02_build_dataset.py          # cleaning, labels, candidates, features (~1 min)
python scripts/03_eda_plots.py              # EDA figures + Folium EDA maps
python scripts/04_train_models.py           # 7 models, 5-fold GroupKFold + student-test (~15 min)
python scripts/05_compare_models.py         # comparison table + figures
python scripts/06_optuna.py 40 hist_gbm     # Bayesian optimisation (TPE)            (~1 h)
python -c "import importlib.util as u,json,sys;s=u.spec_from_file_location('t','scripts/04_train_models.py');m=u.module_from_spec(s);s.loader.exec_module(m);m.run('hist_gbm',json.load(open('experiments/optuna/best_params.json'))['best_params'],'hist_gbm_tuned')"
python scripts/05_compare_models.py         # re-run to include the tuned model
python scripts/07_hmm.py hist_gbm_tuned     # decoder tuning (train OOF) + classifier vs HMM
python scripts/08_final_maps.py             # final Folium maps
python scripts/10_analysis_figures.py       # Optuna/HMM/error-analysis figures
python reports/build_report.py              # PDF report
```

## Running the system

### Laptop 1 – backend

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test      # folder with the 10 evaluation trip folders
deployment\start_backend.bat                          # = python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

On Linux/macOS use `PATHSNAP_EVAL_DIR=/path ./deployment/start_backend.sh`. If `PATHSNAP_EVAL_DIR` is not set, the 10 student-testing
trips are served as a stand-in. Other settings are listed in `deployment/backend.env.example`: `PATHSNAP_CORS_ORIGINS`, `PATHSNAP_MODEL` and `PATHSNAP_HMM_CONFIG`.
Interactive API docs: `http://<LAPTOP1_IP>:8000/docs`. Every request is logged to the console and to `experiments/logs/api.log`.

Allow inbound connections on port 8000. On Windows, run this in an administrator terminal:

```bash
netsh advfirewall firewall add rule name=PathSnap dir=in action=allow protocol=TCP localport=8000
```

### Laptop 2 – frontend

```bash
python deployment/check_api.py http://<LAPTOP1_IP>:8000     # 1) verify the network path
deployment\start_frontend.bat <LAPTOP1_IP>                  # 2) writes VITE_API_BASE_URL, builds, serves on :5173
```

Open `http://localhost:5173`. You can also change the API URL at run time in the dashboard header, then press **Connect**.
For development, use `cd frontend && npm run dev`.

### Networking checklist
1. Both laptops are on the same Wi-Fi/LAN (guest networks with client isolation will block traffic).
2. Find Laptop 1's IPv4 address with `ipconfig` (Windows) or `hostname -I` (Linux).
3. Allow port 8000 through Laptop 1's firewall.
4. CORS allows any origin by default. To restrict it, set `PATHSNAP_CORS_ORIGINS=http://<LAPTOP2_IP>:5173`.
5. `check_api.py` must print `/api/health 200` and the list of trips before the UI is opened.

## Tests

```bash
python -m pytest -q                     # unit + API integration tests (uses the trained artifacts)
```

To run the API test on all ten trips, set `PATHSNAP_ALL_TRIPS=1`. To test the running server over HTTP, use `python scripts/11_e2e_api.py http://localhost:8000`.

## Data-leakage safeguards
- Evaluator data was not part of the supplied material and is never read by any training or tuning script.
- Learned transforms (scalers) sit inside model pipelines and are fitted on training folds only. GroupKFold keeps whole trips together.
- Optuna and HMM decoder tuning use training trips only. HMM tuning uses out-of-fold probabilities.
- The inference path (`MapMatcher.match`) receives only the track and the network. The `.route` file is read only by the server-side evaluator after the
  prediction is final, and only aggregate metrics are returned. A test asserts that no reference fields appear in API responses.

## AI assistance
An AI coding assistant helped write code and documentation. Every reported number was produced by running the scripts in this repository.
The team must review and be able to explain all of the code and results.
