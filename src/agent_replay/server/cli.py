import argparse
import importlib
import sqlite3
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import sys

from agent_replay.server.api import app
from langgraph.checkpoint.sqlite import SqliteSaver

def parse_app_string(app_str: str):
    """Parses module_path:attribute or module_path.attribute"""
    if ":" in app_str:
        module_path, attr = app_str.split(":", 1)
    elif "." in app_str:
        module_path, attr = app_str.rsplit(".", 1)
    else:
        raise ValueError("Invalid app string format. Use module:graph or module.graph")
        
    # Add current directory to path if not there
    if "" not in sys.path and "." not in sys.path:
        sys.path.insert(0, "")
        
    module = importlib.import_module(module_path)
    graph = getattr(module, attr)
    
    # If it's a function (like build_graph()), call it
    if callable(graph) and not hasattr(graph, "invoke"):
        graph = graph()
        
    return graph

def main():
    parser = argparse.ArgumentParser(description="Agent Replay Debugger")
    parser.add_argument("db", help="Path to the trace SQLite database (e.g. trace.sqlite)")
    parser.add_argument("--app", required=True, help="Import path to the compiled LangGraph (e.g. my_agent:graph)")
    parser.add_argument("--port", type=int, default=7337, help="Port to serve on")
    
    args = parser.parse_args()
    
    # 1. Load the Graph
    print(f"Loading graph from {args.app}...")
    try:
        graph = parse_app_string(args.app)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        sys.exit(1)
        
    # 2. Setup SQLite connection
    print(f"Connecting to {args.db}...")
    conn = sqlite3.connect(args.db, check_same_thread=False)
    
    # Ensure LangGraph tables exist
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    
    # We must ensure the graph uses THIS checkpointer so update_state works
    # If the user passed a compiled graph, its checkpointer might be different or None
    # We ideally want the graph to use this connection. Let's mutate it safely if possible,
    # or just assume the user compiled it with checkpointer=SqliteSaver(conn) inside their code.
    # To be safe and ensure the API works:
    app.state.db_conn = conn
    app.state.graph = graph
    app.state.checkpointer = checkpointer
    
    # Note: If the user compiled the graph with a different checkpointer, branch replay might fail
    # or save to a different DB. We'll recommend they compile their graph dynamically or pass it.
    
    print(f"Starting Replay Debugger on http://localhost:{args.port}...")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
