"""Connectivity check from Laptop 2: python deployment/check_api.py http://<LAPTOP1_IP>:8000"""
import json
import sys
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
for path in ("/api/health", "/api/trips"):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        body = json.loads(r.read())
        print(path, r.status, json.dumps(body)[:300])
