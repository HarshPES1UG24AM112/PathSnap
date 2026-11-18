"""Integration tests: FastAPI endpoints with the real trained model on the configured evaluation trips."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "hmm_config.json").exists(), reason="model artifacts not built")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

FORBIDDEN_KEYS = {"route_ref", "reference", "reference_route", "ref_arc", "ground_truth", "label", "labels"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(o, list):
        for v in o[:3]:
            yield from _walk_keys(v)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["model_loaded"] and r.json()["n_eval_trips"] == 10


def test_trips_lists_ten(client):
    r = client.get("/api/trips")
    assert r.status_code == 200
    assert len(r.json()["trips"]) == 10


def test_unknown_and_malformed_trip(client):
    assert client.post("/api/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.post("/api/predict", json={"trip_id": "../../etc"}).status_code == 422


@pytest.mark.parametrize("trip_id", ["00000052"] if os.environ.get("PATHSNAP_ALL_TRIPS") is None else
                         sorted(p.name for p in (ROOT / "data/raw/student_test").iterdir()))
def test_predict_contract_and_no_leakage(client, trip_id):
    r = client.post("/api/predict", json={"trip_id": trip_id})
    assert r.status_code == 200, r.text
    p = r.json()
    assert len(p["predicted_arcs"]) == p["n_observations"] == len(p["gps"])
    assert len(p["route"]) == len(p["route_arc_ids"]) > 0
    assert 0.0 <= p["metrics"]["hmm_top1_accuracy"] <= 1.0
    assert not (FORBIDDEN_KEYS & set(_walk_keys(p)))  # no reference labels / geometry in the response
    m = client.get(f"/api/trips/{trip_id}/metrics").json()
    assert m == p["metrics"]
    net = client.get(f"/api/trips/{trip_id}/network").json()
    assert net["n_arcs_returned"] > 0
