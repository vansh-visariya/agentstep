import argparse
import importlib
import os
import sqlite3
import sys
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from agentstep.server.api import app
from agentstep.sdk.tracer import setup_otel
from langgraph.checkpoint.sqlite import SqliteSaver


def _resolve_ui_build() -> Path | None:
    """Walk up from the package dir looking for ui/dist, then try CWD."""
    # The package lives at <repo>/src/agentstep/server/cli.py
    # We want <repo>/ui/dist — three levels up from this file.
    here = Path(__file__).resolve().parent
    candidates = [
        here.parents[3] / "ui" / "dist",   # repo-rooted: src/agentstep/server -> src/agentstep -> src -> repo
        Path.cwd() / "ui" / "dist",
        Path.cwd() / "dist",
    ]
    for p in candidates:
        if (p / "index.html").is_file():
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

    if callable(graph) and not hasattr(graph, "invoke"):
        graph = graph()

    return graph


def _mount_spa(ui_path: Path) -> None:
    """Mount the built React app at /, with SPA fallback to index.html.

    Order matters: StaticFiles is mounted AFTER the FastAPI app already
    registered /api/* routes, but Starlette matches more-specific paths
    first, so API calls still work. The catch-all on / serves real files
    (CSS, JS) and falls back to index.html for client-side routes.
    """
    # Serve static assets (hashed files in /assets/) directly
    app.mount(
        "/assets",
        StaticFiles(directory=str(ui_path / "assets")),
        name="assets",
    )
    app.mount(
        "/favicon.svg",
        StaticFiles(directory=str(ui_path), html=False),
        name="favicon",
    )

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str = ""):
        # If a real file exists in the dist dir (e.g. icons.svg), serve it.
        target = ui_path / full_path
        if full_path and target.is_file():
            return FileResponse(str(target))
        # Otherwise, SPA fallback to index.html.
        return FileResponse(str(ui_path / "index.html"))


def main():
    parser = argparse.ArgumentParser(description="Agent Replay Debugger")
    parser.add_argument("db", help="Path to trace SQLite database (e.g. trace.sqlite)")
    parser.add_argument(
        "--app", required=True,
        help="Import path to the compiled LangGraph (e.g. sample:graph)",
    )
    parser.add_argument("--port", type=int, default=7337, help="Port to serve on")
    parser.add_argument(
        "--no-ui", action="store_true",
        help="Do not serve the bundled UI. Use this when the Vite dev server "
             "is running on :5173 with its own proxy to /api/*.",
    )
    parser.add_argument(
        "--dev-ui", action="store_true",
        help="Backend-only mode for development. Disables the bundled UI. "
             "Start the Vite dev server separately (cd ui && npm run dev).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default 127.0.0.1). Use 0.0.0.0 for LAN access.",
    )

    args = parser.parse_args()

    print(f"Loading graph from {args.app}…")
    try:
        graph = parse_app_string(args.app)
    except Exception as e:
        print(f"Failed to load graph: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {args.db}…")
    conn = sqlite3.connect(args.db, check_same_thread=False)
    setup_otel(args.db)

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    app.state.db_conn = conn
    app.state.db_path = args.db
    app.state.graph = graph
    app.state.checkpointer = checkpointer

    if args.dev_ui:
        args.no_ui = True
        print(
            "Dev UI mode: bundled UI disabled.\n"
            "  Start the Vite dev server in another terminal:\n"
            "    cd ui && npm install && npm run dev\n"
            "  Then open http://localhost:5173"
        )

    if not args.no_ui:
        ui_path = _resolve_ui_build()
        if ui_path is not None:
            print(f"Serving UI from {ui_path}")
            _mount_spa(ui_path)
        else:
            print(
                "WARNING: UI build not found.\n"
                "  Run from ui/: npm run build\n"
                "  Or start the Vite dev server on :5173 and pass --no-ui.",
                file=sys.stderr,
            )

    print(f"Replay Debugger -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()