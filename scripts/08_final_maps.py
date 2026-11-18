"""Final Folium maps for student-testing trips: raw GPS, local road network, reference route,
classifier-only route and classifier+HMM route, produced by the deployed inference pipeline.
Output: results/maps/final_<trip>.html
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap import io  # noqa: E402
from pathsnap.inference import MapMatcher  # noqa: E402
from pathsnap.visualization import trip_map  # noqa: E402

TRIPS = sys.argv[1:] or io.list_trips("student_test")
mm = MapMatcher(ROOT / "artifacts/models/hist_gbm_tuned.joblib", ROOT / "artifacts/hmm_config.json")
for tid in TRIPS:
    res = mm.match(io.load_track(tid, "student_test"), io.load_network(tid, "student_test"), tid)
    route = io.load_route(tid, "student_test").tolist()  # offline report map only (student-test partition)
    m = trip_map(f"Student-test trip {tid}: GPS vs reference vs PathSnap", res.prepared.pn, res.prepared.ct.df,
                 reference=route, predicted=res.route, classifier=res.clf_route)
    m.save(str(ROOT / "results" / "maps" / f"final_{tid}.html"))
    print(tid, "route arcs", len(res.route), "breaks", res.breaks)
