# Agent Step

A time-travel debugger and branch explorer for **[LangGraph](https://langchain-ai.github.io/langgraph/)** agents. Capture every LLM call and tool invocation as a span, browse them in a web timeline, then **branch from any point** — override the output and replay to see how the rest of the graph would behave differently.

Think `pdb` + a REPL for agent workflows, with a SQLite file you can hand to a teammate.

---

## Why you'd use it

When an agent goes off the rails, you usually want to answer one of:

- *"What did the LLM actually say at step 4?"* — captured.
- *"What would have happened if the weather tool returned snow instead of fog?"* — **branch and replay.**
- *"Why did the agent loop?"* — the timeline shows every call with timing and full prompts/outputs.

Without this, you're adding `print()` statements and re-running with a different seed. With this, you replay against the original trace.

---

## Install

```bash
pip install agentstep
```

Or from the repo (development):

```bash
git clone https://github.com/vanshvisariya/agent-replay
cd agent-replay
pip install -e .
```

Requires **Python 3.13+**.

---

## Quick start

This walks through using Agent Replay on a LangGraph agent in your own project.

### 1. Wrap your graph execution

The SDK exposes one thing: `replay_trace`, a context manager that instruments your graph with OpenTelemetry callbacks and writes spans to a SQLite file.

```python
from langgraph.graph import StateGraph, START, END
from agent_replay.sdk.tracer import replay_trace

# build your compiled graph the way you already do
graph = ...

# a thread_id identifies one conversation/run in the trace
config = {"configurable": {"thread_id": "user-42"}}

with replay_trace(config, sqlite_path="trace.sqlite") as cfg:
    for chunk in graph.stream(inputs, cfg, stream_mode="values"):
        print(chunk)
```

That's the entire API surface for instrumentation. The context manager:

1. Sets up an OpenTelemetry tracer pointed at your SQLite file.
2. Injects a callback handler into `config["callbacks"]`.
3. Records every `llm_call` and `tool_call` span with timing, prompts, completions, and outputs.

The original `config` is mutated in place; you don't need to swap it back.

### 2. Launch the debugger

In a terminal:

```bash
replay-debugger trace.sqlite --app my_module:graph
```

- `trace.sqlite` is the file you wrote spans to.
- `--app my_module:graph` is a Python import path to your compiled graph. Three forms work:
  - `my_module:graph` — `graph` is a compiled LangGraph instance.
  - `my_module.graph` — same thing, dotted form.
  - `my_module:make_graph` — `make_graph` is a callable that returns a compiled graph (it gets called at startup).

Open <http://localhost:7337>.

You should see your thread in the left sidebar and a timeline of spans on the right.

### 3. Branch from any span

1. Click any span — the right panel shows the checkpoint, attributes, and full completion.
2. Click **branch from here**.
3. Edit the override output (new tool result or new LLM completion).
4. Click **run_branch**.

The original trace stays intact. The fork becomes a new branch in the timeline, labeled with a small `b0` chip, color-coded so you can tell at a glance which branch you're looking at.

---

## What gets captured

| Span type | What's recorded |
|---|---|
| `llm_call` | prompt, completion, system, input/output token counts, wall time |
| `tool_call` | tool name, input string, output string, wall time |

Every span carries:

- `lg.thread_id` — the LangGraph `thread_id` so spans from one conversation group together.
- `lg.branch_id` — set automatically on spans created during a branch replay, so the debugger can group them separately.

Other graph node executions, sub-graphs, and conditional edges are not yet instrumented as spans — but the checkpoint data is still preserved by LangGraph itself, so branch replay works regardless.

---

## Working example

The repo ships a runnable demo (`sample.py`) with a fake LLM so you don't need any API keys:

```bash
git clone https://github.com/vanshvisariya/replay
cd agent-replay
pip install -e .
python sample.py                              # writes trace.sqlite
replay-debugger trace.sqlite --app sample:graph
```

Then open <http://localhost:7337>. Click the LLM call → click **branch from here** → change the response → watch the timeline fork.

---

## Development workflow

When hacking on the debugger itself, run the backend and frontend with hot reload:

```bash
# Terminal 1 — backend on :7337, API only
replay-debugger trace.sqlite --app sample:graph --dev-ui

# Terminal 2 — Vite dev server on :5173 (proxies /api/* to :7337)
cd ui
npm install
npm run dev
```

Open <http://localhost:5173> instead. Edits to React files hot-reload; backend edits need a restart.

---

## Programmatic branch replay

The web UI is the main way to branch, but the same operation is available as a function for scripted use:

```python
from agent_replay.server.replayer import replay_branch

result = replay_branch(
    thread_id="user-42",
    checkpoint_id="1efb...",          # from GET /api/traces/{tid}/checkpoints
    node_name="tools",                 # or "agent"
    span_type="tool_call",             # or "llm_call"
    tool_call_id="get_weather",        # tool name for tool spans
    new_output="It's snowing in SF.",
    db_path="trace.sqlite",
)
print(result)  # branch_id of the new replay
```

Useful for regression tests, CI, or batch-exploration of failure modes.

---

## API reference

The FastAPI server (started by the `replay-debugger` CLI) exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/threads` | List all thread IDs in the database. |
| `GET` | `/api/traces/{thread_id}` | All spans for a thread, grouped by branch. |
| `GET` | `/api/traces/{thread_id}/checkpoints` | All checkpoints for a thread. |
| `POST` | `/api/branch` | Fork the graph from a checkpoint with an overridden output. |

`POST /api/branch` body:

```json
{
  "thread_id": "user-42",
  "checkpoint_id": "1efb...",
  "node_name": "agent",
  "span_type": "llm_call",
  "tool_call_id": null,
  "new_output": "The weather is sunny and 72°F."
}
```

Response: `{"branch_id": "branch_a1b2c3...", "status": "ok"}`.

---

## Where things live in your file

After running the demo once:

```
trace.sqlite
├── spans table        ← every llm_call / tool_call, with start/end nanoseconds + JSON attributes
├── checkpoints table  ← LangGraph state snapshots (one per node execution)
└── thread metadata    ← implicit, keyed off lg.thread_id in span attributes
```

Everything is one file. Copy it, share it, commit it for reproduction.

---

## Limitations

- **Python 3.13+ only** — pinned in `pyproject.toml`.
- **LangGraph checkpointers must use SQLite** — `SqliteSaver` is the only supported backend currently; the branch endpoint reads from the same file the tracer wrote to.
- **No remote export** — spans stay local. (The exporter is OpenTelemetry-native, so wiring Jaeger/Zipkin out the side is doable but not built in.)
- **Two span types** — only LLM and tool calls. If you want full graph-node tracing, file an issue.

---

## Contributing

```bash
git clone https://github.com/vanshvisariya/agent-replay
cd agent-replay
pip install -e .
cd ui && npm install
```

Layout:

```
src/agent_replay/
├── sdk/
│   ├── tracer.py        ← replay_trace() + ReplayCallbackHandler
│   └── exporter.py      ← OTel span exporter → SQLite
└── server/
    ├── api.py           ← FastAPI endpoints
    ├── replayer.py      ← branch replay logic (used by API + programmatic)
    └── cli.py           ← `replay-debugger` entry point
ui/
└── src/App.tsx          ← single-file React app
sample.py                ← runnable weather-agent demo
```

---

## License

MIT — see [LICENSE](./LICENSE).