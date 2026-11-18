"""Capture PNG screenshots of Folium maps and of the running UI with Playwright (headless Chromium).

Usage: python scripts/09_screenshots.py maps
       python scripts/09_screenshots.py ui http://localhost:5173 00000052 00000056
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

mode = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    if mode == "maps":
        for f in sorted((ROOT / "results" / "maps").glob("*.html")):
            pg.goto(f.as_uri())
            pg.wait_for_timeout(3500)
            pg.screenshot(path=str(OUT / f"map_{f.stem}.png"))
            print("saved", f.stem)
    else:
        url, trips = sys.argv[2], sys.argv[3:]
        pg.goto(url)
        pg.wait_for_selector(".trip", timeout=30000)
        pg.screenshot(path=str(OUT / "ui_home.png"))
        for t in trips:
            pg.click(f"text=Trip {t}")
            pg.wait_for_selector(".big-num", timeout=180000)
            pg.wait_for_timeout(4000)
            pg.screenshot(path=str(OUT / f"ui_trip_{t}.png"))
            print("saved ui", t)
        # error state: point UI to a dead backend
        pg.fill("#api", "http://127.0.0.1:9")
        pg.click("text=Connect")
        pg.wait_for_selector(".alert", timeout=20000)
        pg.screenshot(path=str(OUT / "ui_error_state.png"))
        pg.fill("#api", "http://localhost:8000")
        pg.click("text=Connect")
    b.close()
