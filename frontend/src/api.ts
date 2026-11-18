import type { Health, NetworkGeometry, Prediction, TripList } from "./types";

const KEY = "pathsnap.apiBase";

export function getApiBase(): string {
  try {
    const saved = window.localStorage.getItem(KEY);
    if (saved) return saved;
  } catch {
    /* storage unavailable */
  }
  return (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";
}

export function setApiBase(url: string) {
  try {
    window.localStorage.setItem(KEY, url.replace(/\/+$/, ""));
  } catch {
    /* ignore */
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 120000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${getApiBase()}${path}`, { ...init, signal: ctrl.signal });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        /* non-JSON error */
      }
      throw new Error(`API ${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw new Error("Request timed out");
    if (e instanceof TypeError) throw new Error(`Cannot reach the PathSnap API at ${getApiBase()} (network/CORS error)`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request<Health>("/api/health", undefined, 8000),
  trips: () => request<TripList>("/api/trips"),
  predict: (trip_id: string) =>
    request<Prediction>("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id }),
    }),
  network: (trip_id: string) => request<NetworkGeometry>(`/api/trips/${trip_id}/network`),
};
