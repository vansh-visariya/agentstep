import sqlite3
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from langchain_core.messages import ToolMessage
from agent_replay.server.replayer import replay_branch

app = FastAPI(title="Agent Replay Debugger")

class BranchRequest(BaseModel):
    thread_id: str
    checkpoint_id: str
    node_name: str
    tool_call_id: str
    new_output: str

@app.get("/api/threads")
def list_threads(request: Request):
    """List all unique thread IDs from OTel spans and LangGraph checkpoints."""
    conn: sqlite3.Connection = request.app.state.db_conn
    cursor = conn.cursor()
    # Fetch threads from OTel spans
    cursor.execute("SELECT DISTINCT thread_id FROM otel_spans WHERE thread_id IS NOT NULL")
    threads = [row[0] for row in cursor.fetchall()]
    return {"threads": threads}

@app.get("/api/traces/{thread_id}")
def get_traces(thread_id: str, request: Request):
    """Fetch OTel spans and LangGraph checkpoints for a thread."""
    conn: sqlite3.Connection = request.app.state.db_conn
    cursor = conn.cursor()
    
    # Fetch spans
    cursor.execute("""
        SELECT span_id, parent_span_id, name, start_time, end_time, attributes, events, status_code 
        FROM otel_spans 
        WHERE thread_id = ? 
        ORDER BY start_time ASC
    """, (thread_id,))
    
    spans = []
    for row in cursor.fetchall():
        spans.append({
            "span_id": row[0],
            "parent_span_id": row[1],
            "name": row[2],
            "start_time": row[3],
            "end_time": row[4],
            "attributes": json.loads(row[5]) if row[5] else {},
            "events": json.loads(row[6]) if row[6] else [],
            "status_code": row[7]
        })
        
    # Fetch checkpoints history using LangGraph
    graph = request.app.state.graph
    if not graph:
        return {"spans": spans, "checkpoints": []}
        
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    
    checkpoints = []
    for state in history:
        checkpoints.append({
            "checkpoint_id": state.config["configurable"]["checkpoint_id"],
            "next": state.next,
            # We skip values here to avoid huge payloads, can add a separate endpoint if needed
        })
        
    return {"spans": spans, "checkpoints": checkpoints}

@app.post("/api/branch")
def branch_replay(req: BranchRequest, request: Request):
    """Fork execution with modified tool output."""
    graph = request.app.state.graph
    if not graph:
        raise HTTPException(status_code=400, detail="Graph not loaded")
        
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "checkpoint_id": req.checkpoint_id
        }
    }
    
    # Construct the overridden message
    # In LangGraph, to update a specific message, you typically provide it with its ID.
    # Alternatively, for ToolMessages, providing the tool_call_id updates it if the reducer matches by tool_call_id or id.
    overridden_msg = ToolMessage(
        content=req.new_output,
        tool_call_id=req.tool_call_id
    )
    
    try:
        new_state = replay_branch(
            graph=graph,
            config=config,
            node_name=req.node_name,
            new_values={"messages": [overridden_msg]}
        )
        return {"status": "success", "new_thread_id": req.thread_id} # LangGraph branch retains thread_id but adds a new checkpoint fork, or creates new thread if specified
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
