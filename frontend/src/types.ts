export type LatLng = [number, number];

export interface Health {
  status: string;
  model: string;
  model_loaded: boolean;
  hmm_params: Record<string, number>;
  eval_partition: string;
  n_eval_trips: number;
}

export interface TripInfo {
  trip_id: string;
  n_observations: number;
  duration_s: number;
  n_nodes: number;
  n_arcs: number;
}

export interface TripList {
  partition: string;
  trips: TripInfo[];
}

export interface Metrics {
  hmm_top1_accuracy: number;
  classifier_top1_accuracy: number;
  hmm_arc_macro_f1: number;
  classifier_arc_macro_f1: number;
  hmm_route_continuity: number;
  classifier_route_continuity: number;
  hmm_route_precision: number;
  hmm_route_recall: number;
  hmm_route_f1: number;
  candidate_miss_rate: number;
  n_evaluated_observations: number;
}

export interface Prediction {
  trip_id: string;
  partition: string;
  n_observations: number;
  n_dropped_observations: number;
  gps: LatLng[];
  predicted_arcs: number[];
  classifier_arcs: number[];
  route_arc_ids: number[];
  route: LatLng[][];
  classifier_route: LatLng[][];
  unresolved_breaks: number;
  mean_candidates: number;
  timings_ms: Record<string, number>;
  metrics: Metrics | null;
  model: string;
  hmm_params: Record<string, number>;
}

export interface NetworkGeometry {
  trip_id: string;
  n_arcs_total: number;
  n_arcs_returned: number;
  arcs: LatLng[][];
}
