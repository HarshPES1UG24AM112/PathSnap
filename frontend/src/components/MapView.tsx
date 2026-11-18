import { useEffect, useMemo } from "react";
import L from "leaflet";
import { CircleMarker, LayerGroup, LayersControl, MapContainer, Polyline, TileLayer, useMap } from "react-leaflet";
import type { LatLng, NetworkGeometry, Prediction } from "../types";

export const COLORS = {
  gps: "#2b7bff",
  route: "#e5322d",
  classifier: "#f59e0b",
  network: "#8b95a5",
};

function FitBounds({ points }: { points: LatLng[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [30, 30] });
  }, [points, map]);
  return null;
}

interface Props {
  prediction: Prediction | null;
  network: NetworkGeometry | null;
}

export default function MapView({ prediction, network }: Props) {
  const gps = prediction?.gps ?? [];
  // thin very long tracks for rendering only (all points are still used by the API)
  const gpsShown = useMemo(() => {
    const step = Math.max(1, Math.floor(gps.length / 4000));
    return gps.filter((_, i) => i % step === 0);
  }, [gps]);

  return (
    <div className="map-wrap">
      <MapContainer center={[48.0, 8.0]} zoom={4} preferCanvas className="map">
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Blank (offline)">
            <TileLayer url="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" />
          </LayersControl.BaseLayer>
          <LayersControl.Overlay checked name="Road network">
            <LayerGroup>{network && <Polyline positions={network.arcs} pathOptions={{ color: COLORS.network, weight: 1, opacity: 0.55 }} />}</LayerGroup>
          </LayersControl.Overlay>
          <LayersControl.Overlay name="Classifier-only route (no HMM)">
            <LayerGroup>
              {prediction && <Polyline positions={prediction.classifier_route} pathOptions={{ color: COLORS.classifier, weight: 3, opacity: 0.85 }} />}
            </LayerGroup>
          </LayersControl.Overlay>
          <LayersControl.Overlay checked name="Reconstructed route (classifier + HMM)">
            <LayerGroup>{prediction && <Polyline positions={prediction.route} pathOptions={{ color: COLORS.route, weight: 4, opacity: 0.9 }} />}</LayerGroup>
          </LayersControl.Overlay>
          <LayersControl.Overlay checked name="Raw GPS observations">
            <LayerGroup>
              {gpsShown.map((p, i) => (
                <CircleMarker key={i} center={p} radius={2.5} pathOptions={{ color: COLORS.gps, weight: 0, fillOpacity: 0.75 }} />
              ))}
            </LayerGroup>
          </LayersControl.Overlay>
        </LayersControl>
        <FitBounds points={gps} />
      </MapContainer>
      <div className="legend">
        <div><span className="sw dot" style={{ background: COLORS.gps }} />Raw GPS observations</div>
        <div><span className="sw" style={{ background: COLORS.route }} />Reconstructed route (HMM)</div>
        <div><span className="sw" style={{ background: COLORS.classifier }} />Classifier-only (toggle in layers)</div>
        <div><span className="sw thin" style={{ background: COLORS.network }} />Road network</div>
      </div>
      {!prediction && <div className="map-empty">Select an evaluation trip to run the map-matching pipeline</div>}
    </div>
  );
}
