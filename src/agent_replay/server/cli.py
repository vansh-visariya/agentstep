import argparse
import importlib
import os
import sqlite3
import sys

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from agent_replay.server.api import app
from agent_replay.sdk.tracer import setup_otel
from langgraph.checkpoint.sqlite import SqliteSaver


def _resolve_ui_build() -> str | None:
    """Walk up from the package dir looking for ui/dist, then try CWD."""
    candidates = [
        # relative to this file's package
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ui", "dist")
        ),
        # relative to CWD
        os.path.abspath("ui/dist"),
        # relative to CWD (standalone dist)
        os.path.abspath("dist"),
    ]
    for p in candidates:
        index = os.path.join(p, "index.html")
        if os.path.isfile(index):
            return p
    return None


def parse_app_string(app_str: str):
    """Import a LangGraph from ``module:graph`` or ``module.graph``."""
    if ":" in app_str:
        module_path, attr = app_str.split(":", 1)
    elif "." in app_str:
        module_path, attr = app_str.rsplit(".", 1)
    else:
        raise ValueError("Use module:graph or module.graph syntax")

    if "" not in sys.path and "." not in sys.path:
        sys.path.insert(0, "")

    module = importlib.import_module(module_path)
    graph = getattr(module, attr)

    # If it's a factory function call it
    if callable(graph) and not hasattr(graph, "invoke"):
        graph = graph()

    return graph


def main():
    parser = argparse.ArgumentParser(description="Agent Replay Debugger")
    parser.add_argument("db", help="Path to trace SQLite database (e.g. trace.sqlite)")
    parser.add_argument(
        "--app", required=True,
        help="Import path to the compiled LangGraph (e.g. sample:graph)",
    )
    parser.add_argument("--port", type=int, default=7337, help="Port to serve on")
    parser.add_argument(
        "--dev-ui", action="store_true",
        help="Skip serving static UI (use Vite dev proxy instead)",
    )

    args = parser.parse_args()

    # ── Load graph ─────────────────────────────────────────────
    print(f"Loading graph from {args.app}…")
    try:
        graph = parse_app_string(args.app)
    except Exception as e:
        print(f"Failed to load graph: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Setup DB ───────────────────────────────────────────────
    print(f"Connecting to {args.db}…")
    conn = sqlite3.connect(args.db, check_same_thread=False)
    setup_otel(args.db)

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    app.state.db_conn = conn
    app.state.db_path = args.db
    app.state.graph = graph
    app.state.checkpointer = checkpointer

    # ── Serve UI ───────────────────────────────────────────────
    if not args.dev_ui:
        ui_path = _resolve_ui_build()
        if ui_path is not None:
            print(f"Serving UI from {ui_path}")
            app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
        else:
            print(
                "WARNING: UI build not found. Run in ui/: npm run build\n"
                "  Or pass --dev-ui to proxy to Vite dev server on port 5173.",
                file=sys.stderr,
            )

    print(f"Replay Debugger -> http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
