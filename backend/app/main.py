"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .schemas import Health, Metrics, NetworkGeometry, PredictRequest, Prediction, TripInfo, TripList
from .service import PathSnapService, TripNotFound

config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_FILE, encoding="utf-8")])
log = logging.getLogger("pathsnap.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = PathSnapService()
    log.info("PathSnap API ready - %d evaluation trips available", len(state["svc"].trip_ids()))
    yield


app = FastAPI(title="PathSnap API", version="1.0.0", description="GPS map matching: classifier + HMM/Viterbi", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    client = request.client.host if request.client else "-"
    log.info("%s %s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.url.query or "", client, resp.status_code, (time.perf_counter() - t0) * 1000)
    return resp


def svc() -> PathSnapService:
    return state["svc"]


def _not_found(tid):
    raise HTTPException(status_code=404, detail=f"Trip '{tid}' is not one of the available evaluation trips")


@app.get("/api/health", response_model=Health)
def health():
    s = svc()
    return Health(status="ok", model=config.MODEL_PATH.name, model_loaded=s.matcher.model is not None,
                  hmm_params=s.matcher.hmm.__dict__, eval_partition=config.EVAL_LABEL, n_eval_trips=len(s.trip_ids()))


@app.get("/api/trips", response_model=TripList)
def trips():
    s = svc()
    return TripList(partition=config.EVAL_LABEL, trips=[TripInfo(**s.trip_info(t)) for t in s.trip_ids()])


def _prediction(trip_id: str) -> Prediction:
    s = svc()
    try:
        res, metrics = s.predict(trip_id)
    except TripNotFound:
        _not_found(trip_id)
    except Exception as e:  # pragma: no cover - surfaced to client, logged with traceback
        log.exception("prediction failed for %s", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")
    pt = res.prepared
    df = pt.ct.df
    return Prediction(
        trip_id=trip_id, partition=config.EVAL_LABEL, n_observations=len(df), n_dropped_observations=pt.ct.n_dropped,
        gps=[[round(a, 6), round(b, 6)] for a, b in zip(df.lat.tolist(), df.lon.tolist())],
        predicted_arcs=res.hmm_arcs.tolist(), classifier_arcs=res.clf_arcs.tolist(), route_arc_ids=res.route,
        route=s.polylines(res, res.route), classifier_route=s.polylines(res, res.clf_route), unresolved_breaks=res.breaks,
        mean_candidates=float(pt.X.groupby("obs").size().mean()) if len(pt.X) else 0.0,
        timings_ms={k: round(v, 1) for k, v in res.timings_ms.items()},
        metrics=Metrics(**metrics) if metrics else None, model=config.MODEL_PATH.name, hmm_params=s.matcher.hmm.__dict__)


@app.post("/api/predict", response_model=Prediction)
def predict(req: PredictRequest):
    log.info("prediction request for trip %s", req.trip_id)
    return _prediction(req.trip_id)


@app.get("/api/trips/{trip_id}/route", response_model=Prediction)
def route(trip_id: str):
    return _prediction(trip_id)


@app.get("/api/trips/{trip_id}/metrics", response_model=Metrics)
def metrics(trip_id: str):
    p = _prediction(trip_id)
    if p.metrics is None:
        raise HTTPException(status_code=404, detail="No server-side reference available for this trip")
    return p.metrics


@app.get("/api/trips/{trip_id}/network", response_model=NetworkGeometry)
def network(trip_id: str):
    try:
        return NetworkGeometry(**svc().network(trip_id))
    except TripNotFound:
        _not_found(trip_id)
