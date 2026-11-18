"""Backend settings (environment variables override defaults; see deployment/backend.env.example)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).resolve() if v else default


# Directory holding the ten evaluation trips (<dir>/<trip_id>/<trip_id>.{track,nodes,arcs,route}).
# On evaluation day point PATHSNAP_EVAL_DIR at the evaluator-testing folder. Until then the
# student-testing partition is used so the full system can be demonstrated end-to-end.
EVAL_DIR = _path("PATHSNAP_EVAL_DIR", ROOT / "data" / "raw" / "student_test")
EVAL_LABEL = os.environ.get("PATHSNAP_EVAL_LABEL", "student_test (stand-in for evaluator trips)")
MODEL_PATH = _path("PATHSNAP_MODEL", ROOT / "artifacts" / "models" / "hist_gbm_tuned.joblib")
HMM_CONFIG = _path("PATHSNAP_HMM_CONFIG", ROOT / "artifacts" / "hmm_config.json")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
MAX_NETWORK_ARCS = int(os.environ.get("PATHSNAP_MAX_NETWORK_ARCS", "25000"))
LOG_FILE = _path("PATHSNAP_LOG_FILE", ROOT / "experiments" / "logs" / "api.log")
