#!/bin/bash
# Build for Render: install deps, build React, copy to backend/static
set -e
pip install -r requirements.txt
npm install
# Empty API URL = use same origin (frontend served from backend)
REACT_APP_API_URL= npm run build
mkdir -p backend/static
cp -r build/* backend/static/
