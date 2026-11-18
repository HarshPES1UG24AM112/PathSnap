from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    model: str
    model_loaded: bool
    hmm_params: dict
    eval_partition: str
    n_eval_trips: int


class TripInfo(BaseModel):
    trip_id: str
    n_observations: int
    duration_s: float
    n_nodes: int
    n_arcs: int


class TripList(BaseModel):
    partition: str
    trips: list[TripInfo]


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^\d{1,12}$", examples=["00000052"])


class Metrics(BaseModel):
    """Computed server-side against reference labels that never leave the server."""
    hmm_top1_accuracy: float
    classifier_top1_accuracy: float
    hmm_arc_macro_f1: float
    classifier_arc_macro_f1: float
    hmm_route_continuity: float
    classifier_route_continuity: float
    hmm_route_precision: float
    hmm_route_recall: float
    hmm_route_f1: float
    candidate_miss_rate: float
    n_evaluated_observations: int


class Prediction(BaseModel):
    trip_id: str
    partition: str
    n_observations: int
    n_dropped_observations: int
    gps: list[list[float]]  # [lat, lon] of cleaned observations, in time order
    predicted_arcs: list[int]  # HMM arc id per observation (-1 = none)
    classifier_arcs: list[int]  # classifier-only arg-max arc per observation
    route_arc_ids: list[int]  # reconstructed connected route (arc ids)
    route: list[list[list[float]]]  # polylines [[lat,lon],[lat,lon]] per route arc
    classifier_route: list[list[list[float]]]
    unresolved_breaks: int
    mean_candidates: float
    timings_ms: dict
    metrics: Metrics | None
    model: str
    hmm_params: dict


class NetworkGeometry(BaseModel):
    trip_id: str
    n_arcs_total: int
    n_arcs_returned: int
    arcs: list[list[list[float]]]
