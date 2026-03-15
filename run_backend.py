#!/usr/bin/env python3
"""Entry point for Render when using project root. Runs backend."""
import os
import sys

# Change to backend directory so uvicorn finds main:app and analysis
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
os.chdir(backend_dir)
sys.path.insert(0, backend_dir)

import uvicorn

port = int(os.environ.get("PORT", "8000"))
uvicorn.run("main:app", host="0.0.0.0", port=port)
