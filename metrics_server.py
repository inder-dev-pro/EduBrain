"""
metrics_server.py
-----------------
Runs a lightweight HTTP server on port 9090 to expose Prometheus metrics.
This runs as a background daemon thread alongside the Streamlit app.
"""

import threading
from wsgiref.simple_server import make_server, WSGIRequestHandler
from prometheus_client import make_wsgi_app


class _SilentHandler(WSGIRequestHandler):
    """Suppress default access-log noise in Streamlit's stdout."""

    def log_message(self, format, *args):  # noqa: A002
        pass


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus metrics HTTP server in a background daemon thread."""
    app = make_wsgi_app()
    httpd = make_server("", port, app, handler_class=_SilentHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
