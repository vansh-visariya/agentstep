import sqlite3
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agent_replay.server.replayer import replay_branch
from agent_replay.sdk.tracer import replay_trace

from langchain_core.messages import ToolMessage, AIMessage

app = FastAPI(title="Agent Replay Debugger")


class BranchRequest(BaseModel):
    thread_id: str
    checkpoint_id: str
    node_name: str
    span_type: str
    tool_call_id: Optional[str] = None  # tool *name* sent by the UI
    new_output: str


# ── Threads ────────────────────────────────────────────────────

@app.get("/api/threads")
def list_threads(request: Request):
    conn: sqlite3.Connection = request.app.state.db_conn
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM otel_spans WHERE thread_id IS NOT NULL")
    return {"threads": [row[0] for row in cursor.fetchall()]}


# ── Traces ─────────────────────────────────────────────────────

@app.get("/api/traces/{thread_id}")
def get_traces(thread_id: str, request: Request):
    conn: sqlite3.Connection = request.app.state.db_conn
    cursor = conn.cursor()

    cursor.execute("""
        SELECT span_id, parent_span_id, name, start_time, end_time,
               attributes, events, status_code
        FROM otel_spans
        WHERE thread_id = ?
        ORDER BY start_time ASC
    """, (thread_id,))

    spans = [
        {
            "span_id": r[0],
            "parent_span_id": r[1],
            "name": r[2],
            "start_time": r[3],
            "end_time": r[4],
            "attributes": json.loads(r[5]) if r[5] else {},
            "events": json.loads(r[6]) if r[6] else [],
            "status_code": r[7],
        }
        for r in cursor.fetchall()
    ]

    return {"spans": spans}


# ── Checkpoints ────────────────────────────────────────────────

@app.get("/api/traces/{thread_id}/checkpoints")
def get_checkpoints(thread_id: str, request: Request):
    """Return checkpoints with their *next* node info for matching to spans."""
    graph = request.app.state.graph
    if not graph:
        return {"checkpoints": []}

    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))

    checkpoints = []
    for state in history:
        cp = {
            "checkpoint_id": state.config["configurable"]["checkpoint_id"],
            "next": list(state.next) if state.next else [],
            "has_messages": bool(state.values.get("messages")),
        }
        checkpoints.append(cp)

    return {"checkpoints": checkpoints}


# ── Branch replay ──────────────────────────────────────────────

@app.post("/api/branch")
def branch_replay(req: BranchRequest, request: Request):
    graph = request.app.state.graph
    if not graph:
        raise HTTPException(400, "Graph not loaded — pass --app <module:graph>")

    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "checkpoint_id": req.checkpoint_id,
        }
    }

    # Sanity-check the checkpoint exists and extract the state
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        raise HTTPException(400, f"Checkpoint {req.checkpoint_id} not found: {e}")

    full_config = snapshot.config

    # Ensure checkpoint_ns is present — LangGraph requires it when
    # resuming from a checkpoint.
    full_config.setdefault("configurable", {})
    if "checkpoint_ns" not in full_config["configurable"]:
        full_config["configurable"]["checkpoint_ns"] = ""

    # ── Resolve real tool_call_id ──────────────────────────────
    # The UI sends the tool *name* (e.g. "get_weather") as
    # tool_call_id.  We need to find the actual ID from the
    # AIMessage tool_calls in the checkpoint state.
    if req.span_type == "tool_call":
        tool_name = req.tool_call_id or ""
        messages = snapshot.values.get("messages", [])
        real_tool_call_id = None

        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict) and tc.get("name") == tool_name:
                        real_tool_call_id = tc["id"]
                        break
                    elif hasattr(tc, "name") and tc.name == tool_name:
                        real_tool_call_id = tc.id
                        break
                if real_tool_call_id:
                    break

        if not real_tool_call_id:
            raise HTTPException(
                400,
                f"No tool call named '{tool_name}' found in checkpoint state "
                f"(messages: {len(messages)}). Available tool calls: "
                + ", ".join(
                    tc.get("name", tc.name) if isinstance(tc, dict) else tc.name
                    for msg in messages if hasattr(msg, "tool_calls")
                    for tc in (msg.tool_calls or [])
                ),
            )

        overridden_msg = ToolMessage(
            content=req.new_output,
            tool_call_id=real_tool_call_id,
        )
        node = req.node_name or "tools"

    elif req.span_type == "llm_call":
        overridden_msg = AIMessage(content=req.new_output)
        node = req.node_name or "agent"

    else:
        raise HTTPException(400, f"Unsupported span_type: {req.span_type}")

    # ── Replay ─────────────────────────────────────────────────
    try:
        with replay_trace(full_config, sqlite_path=request.app.state.db_path):
            result = replay_branch(
                graph=graph,
                config=full_config,
                node_name=node,
                new_values={"messages": [overridden_msg]},
            )
    except Exception as e:
        raise HTTPException(500, f"Branch replay failed: {e}")

    return {
        "status": "ok",
        "thread_id": req.thread_id,
        "checkpoint_id": full_config["configurable"].get("checkpoint_id"),
    }
