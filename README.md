# Agent Replay

A time-travel debugger and branch explorer for **LangGraph-based AI agents**. Capture an agent's execution trace, visualize every LLM call and tool invocation in a web UI, then **branch from any point** — override the output and replay to see how the rest of the graph behaves differently.

Think of it as a debugger + REPL for agent workflows.

---

## How It Works

```
┌─ Your LangGraph Agent ─────────────────────────────┐
│  with replay_trace(config):                         │
│      graph.stream(inputs, config)                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─ Agent Replay SDK ──────────────────────────────────┐
│  Captures LLM calls & tool invocations as spans     │
│  Writes them to a local SQLite database             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─ Debugger UI (FastAPI + React) ─────────────────────┐
│  Shows a timeline of every LLM & tool call          │
│  Click any span → inspect full prompt/input/output  │
│  Click "Branch" → edit the output → replay          │
│  Original trace is preserved; fork shows as new     │
└─────────────────────────────────────────────────────┘
```

The debugger stores everything in a single SQLite file — the trace data and LangGraph checkpoints live side by side. You can share the file or keep it local.

---

## Features

- **Execution timeline** — See every LLM call and tool invocation with timing, ordered chronologically
- **Full detail inspection** — Click any span to see the raw prompt, completion, tool inputs, and outputs
- **Branch replay** — Fork the agent's state at any point, override the LLM or tool output, and re-run from there
- **One-file persistence** — Everything lives in a portable SQLite database. No servers, no cloud
- **No API key required** — Ships with a fake LLM for testing without any provider credentials
- **OpenTelemetry-native** — Traces use the OTel span format, so you could export to Jaeger, Zipkin, etc.

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 2. Install the package and its dependencies
pip install -e .

# 3. Build the web UI
cd ui
npm install
npm run build
cd ..

# 4. Run the sample agent (generates trace.sqlite)
python sample.py

# 5. Start the debugger
replay-debugger trace.sqlite --app sample:graph

# 6. Open http://localhost:7337
```

---

## Instrumenting Your Own Agent

Wrap your LangGraph execution with the `replay_trace` context manager:

```python
from agent_replay.sdk.tracer import replay_trace

config = {"configurable": {"thread_id": "my-session"}}

with replay_trace(config, sqlite_path="trace.sqlite"):
    for chunk in graph.stream(inputs, config, stream_mode="values"):
        print(chunk)
```

Then start the debugger:

```bash
replay-debugger trace.sqlite --app my_module:graph
```

The `--app` argument takes a Python import path to your compiled LangGraph. It supports `module:graph`, `module.graph`, or a callable factory that returns the compiled graph.

<details>
<summary>Example — callable factory</summary>

```python
# my_agent.py
def make_graph():
    builder = StateGraph(AgentState)
    # ... add nodes and edges ...
    return builder.compile(checkpointer=SqliteSaver.from_conn_string("trace.sqlite"))
```

```bash
replay-debugger trace.sqlite --app my_agent:make_graph
```
</details>

---

## Development Mode (HMR)

Run the backend and frontend side by side for hot reload:

```bash
# Terminal 1 — Backend (API on :7337)
replay-debugger trace.sqlite --app sample:graph --dev-ui

# Terminal 2 — Frontend (Vite dev server on :5173)
cd ui
npm run dev
```

The Vite dev server proxies `/api/*` requests to the FastAPI backend on port 7337.

---

## API Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `GET` | `/api/threads` | List all thread IDs in the database |
| `GET` | `/api/traces/{thread_id}` | Get all spans for a thread, ordered by time |
| `GET` | `/api/traces/{thread_id}/checkpoints` | Get state snapshots for a thread |
| `POST` | `/api/branch` | Fork the graph from a checkpoint with overridden output |

**Branch payload:**

```json
{
  "thread_id": "my-session",
  "checkpoint_id": "1efb...",
  "node_name": "agent",
  "span_type": "llm_call",
  "tool_call_id": "get_weather",
  "new_output": "The weather is sunny and 72°F."
}
```

---

## Project Structure

```
├── src/agent_replay/
│   ├── sdk/            # Instrumentation layer
│   │   ├── tracer.py   # replay_trace() context manager + callback handler
│   │   └── exporter.py # OTel span exporter → SQLite
│   └── server/         # Debugger backend
│       ├── api.py      # FastAPI endpoints
│       ├── replayer.py # Branch replay logic
│       └── cli.py      # replay-debugger CLI
├── ui/                 # React frontend
│   └── src/
│       └── App.tsx     # All UI logic (single file)
├── sample.py           # Weather agent demo (no API key needed)
├── pyproject.toml      # Python package definition
└── trace.sqlite        # Generated when you run sample.py
```

---

## What Gets Traced

Currently: **LLM calls** (prompt → completion) and **tool invocations** (input → output). These are the two most common places to debug in an agent loop — you see exactly what the LLM said and what the tools returned.

Other graph node executions, sub-graphs, and conditional edges are not yet traced, but the checkpoint data preserves the full state so branch replay works regardless.

---

## Requirements

- **Python 3.13+**
- **Node.js** (to build the UI)
- Works on Windows, macOS, and Linux
