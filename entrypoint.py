#!/usr/bin/env python3
"""
entrypoint.py
-------------
Starts the Prometheus metrics HTTP server on port 9091 BEFORE launching
Streamlit. This guarantees port 9091 is bound exactly ONCE per container
lifetime — independently of Streamlit's script re-execution model, which
re-runs app.py on every user interaction.
"""

import subprocess
import sys
from metrics_server import start_metrics_server

# Bind port 9091 once — in the parent process, before Streamlit forks/spawns
start_metrics_server(port=9091)
print("[entrypoint] Prometheus metrics server listening on :9091", flush=True)

# Hand off to Streamlit — this blocks until Streamlit exits
subprocess.run(
    ["streamlit", "run", "app.py"] + sys.argv[1:],
    check=True,
)
