#!/usr/bin/env bash
# Laptop 2: build and serve the UI. Usage: ./start_frontend.sh 192.168.1.10
cd "$(dirname "$0")/../frontend"
API="http://${1:-localhost}:8000"
echo "VITE_API_BASE_URL=$API" > .env.local
npm install && npm run build && npm run preview
