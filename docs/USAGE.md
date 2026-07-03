# AgentStep Usage Guide

AgentStep is a time-travel debugger for LangGraph agents. It records execution traces — LLM calls, retriever queries, graph node runs, agent reasoning steps, and tool invocations — into a SQLite file, lets you inspect the run in a web UI, and lets you branch from any step with an overridden output.

## Install

Install from PyPI:

```bash
pip install agentstep
```

Install from a local checkout during development:

```bash
git clone https://github.com/vanshvisariya/agentstep
cd agentstep
pip install -e .
```

AgentStep requires Python 3.13 or newer.

## Instrument your graph

Use the tracing context manager from `agentstep.sdk.tracer` when you run your LangGraph graph. The context manager writes spans to a SQLite file.

```python
from agentstep.sdk.tracer import replay_trace

config = {"configurable": {"thread_id": "user-42"}}

with replay_trace(config, sqlite_path="trace.sqlite") as cfg:
    result = graph.stream(inputs, cfg, stream_mode="values")
    for chunk in result:
        print(chunk)
```

What this does:

1. Hooks OpenTelemetry into the graph run.
2. Stores spans in `trace.sqlite`.
3. Adds the callback handler to the LangGraph config in place.

## What gets traced

AgentStep captures **6 span types** via LangChain/LangGraph callbacks:

### 1. `llm_call` — Legacy LLM inference

Triggered by `on_llm_start/end`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.system` | string | Provider name (e.g. `"langgraph"`) |
| `gen_ai.prompt` | string | First prompt text sent to the model |
| `gen_ai.completion` | string | Generated response text |
| `gen_ai.usage.input_tokens` | int | Input token count |
| `gen_ai.usage.output_tokens` | int | Output token count |

### 2. `chat_model_call` — Modern chat model inference

Triggered by `on_chat_model_start/end`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.system` | string | Provider name (e.g. `"openai"`, `"anthropic"`) |
| `gen_ai.chat.first_message_preview` | string | First message content, truncated to 500 chars |
| `gen_ai.chat.message_count` | int | Total messages across all conversation turns |
| `gen_ai.completion` | string | Generated response text (same as `llm_call`) |
| `gen_ai.usage.input_tokens` | int | Input token count |
| `gen_ai.usage.output_tokens` | int | Output token count |

> **Why two LLM span types?** Modern LangChain routes chat models through `on_chat_model_*` separately from legacy `on_llm_*`. Without this handler, all your chat model calls would be invisible.

### 3. `retriever_call` — RAG retrieval queries

Triggered by `on_retriever_start/end/error`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.system` | string | Retriever ID (e.g. `"langchain_community.retrievers...Chroma"`) |
| `gen_ai.query` | string | The query text sent to the retriever |
| `gen_ai.retriever.document_count` | int | Number of documents returned |
| `gen_ai.retriever.first_doc_preview` | string | First document content, truncated to 500 chars |
| `gen_ai.retriever.metadata_keys` | JSON array | Key names from the first document's metadata (not values) |

> **Design note:** Full retrieved documents are not stored — they would bloat the SQLite file. The preview + metadata keys give enough signal to judge retrieval quality without the cost.

### 4. `node_run` — Graph node execution

Triggered by `on_chain_start/end/error`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.node.name` | string | Node name extracted from `langgraph:chain:<name>` or fallback to serialized name |
| `gen_ai.node.input_keys` | JSON array | Keys present in the node's input state (up to 20) |
| `gen_ai.node.output_keys` | JSON array | Keys present in the node's output state (up to 20) |

> **Why this matters:** This is the graph topology. It tells you which nodes ran, in what order, and how data flowed between them — something `llm_call` + `tool_call` alone cannot show.

### 5. `agent_step` — Agent reasoning decisions

Triggered by `on_agent_action/finish`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.agent.tool` | string | Name of the tool the agent chose to call |
| `gen_ai.agent.tool_input` | string | Tool input text, truncated to 500 chars (or key names if dict) |
| `gen_ai.agent.log_preview` | string | Agent's reasoning trace log, truncated to 300 chars |
| `gen_ai.agent.return_keys` | JSON array | Keys from the agent's final return values (on finish only) |
| `gen_ai.agent.final_log_preview` | string | Final reasoning log, truncated to 300 chars (finish only) |

> **Why this matters:** Shows what the agent *thought* before each tool call — not just which tools it used. Essential for debugging multi-step reasoning loops.

### 6. `tool_call` — Tool invocations (existing)

Triggered by `on_tool_start/end/error`. Captures:

| Attribute | Type | Description |
|---|---|---|
| `gen_ai.tool.name` | string | Tool class name |
| `gen_ai.tool.input` | string | Input passed to the tool |
| `gen_ai.tool.output` | string | Output returned by the tool |

### Span-level attributes (all types)

Every span carries these common attributes:

| Attribute | Type | Description |
|---|---|---|
| `lg.thread_id` | string | LangGraph thread ID for grouping runs |
| `lg.branch_id` | string or absent | Branch identifier — only present on branched replays, absent on the original trace |

## Run the debugger

After you have a trace file, launch the debugger with the `replay-debugger` command:

```bash
replay-debugger trace.sqlite --app sample:graph
```

The `--app` argument should point at your compiled graph:

- `my_module:graph` for a graph object
- `my_module.graph` for dotted import syntax
- `my_module:make_graph` for a callable that returns a compiled graph

Open `http://localhost:7337` in your browser.

### Development UI mode

When editing the frontend or backend, start the server without the bundled UI:

```bash
replay-debugger trace.sqlite --app sample:graph --dev-ui
```

Then run the Vite dev server from `ui/`:

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173` instead.

## Reading traces in the UI

The timeline shows one row per span, ordered by start time within each branch. Each row displays:

- **Icon** — identifies the span type (Play=LLM, Database=retriever, GitBranch=node, Layers=agent step, Code=tool)
- **Type label** — `Tool Call`, `Retriever Call`, `Node Run`, `Agent Step`, `Chat Model Call`, or `LLM Call`
- **Chip tag** — short category: `tool`, `retrieval`, `node`, `step`, `model`, or `agent`
- **Detail text** — type-specific summary (e.g. tool name, query preview, node name)
- **Duration** — wall-clock time for the span

Click any row to open the inspector panel on the right, which shows:

1. **Checkpoint** — which checkpoint this step maps to (thread ID, step ID, next nodes)
2. **Attributes table** — all span attributes with values
3. **Output section** — type-specific output preview (Completion, Tool Output, Retrieval Result, Agent Decision)
4. **Fork Info** — whether this span is a branch point

## Branch from a step

In the UI:

1. Click a span in the timeline.
2. Click **branch from here**.
3. Edit the override output (pre-filled with current value when available).
4. Run the branch replay.

The original trace stays unchanged and the branch is added as a new replay path.

## Programmatic branch replay

For scripted branching, use `replay_branch`:

```python
from agentstep.server.replayer import replay_branch
from langchain_core.messages import AIMessage

# 1. Find the checkpoint you want to fork from (e.g. via GET /api/traces/{tid}/checkpoints).
checkpoint_id = "..."

# 2. Call replay_branch with the graph, config, and overridden state.
result = replay_branch(
    graph=graph,
    config={"configurable": {"thread_id": "user-42", "checkpoint_id": checkpoint_id}},
    node_name="agent",
    new_values={"messages": [AIMessage(content="The weather is sunny.")]},
)

# result contains the final state of the branched execution.
```

The `node_name` determines which graph node resumes from the fork point (e.g. `"tools"`, `"agent"`). The `new_values` dict merges into the checkpoint's state — typically you override a message to inject an alternative tool result or LLM completion.

## API endpoints

The debugger server exposes these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/threads` | List thread ids |
| `GET` | `/api/traces/{thread_id}` | Get spans grouped by branch, with fork point metadata |
| `GET` | `/api/traces/{thread_id}/checkpoints` | Get checkpoints for a thread (next nodes, message presence) |
| `POST` | `/api/branch` | Create a replay branch from a checkpoint |

## Troubleshooting

- If the UI 404s, build the frontend first with `cd ui && npm run build`.
- If port `7337` is already in use, stop the old debugger process or pass `--port`.
- If the debugger cannot find your graph, double-check the import path passed to `--app`.
- If retriever spans are missing from your trace, ensure you're using a LangChain-compatible retriever that emits `on_retriever_start`/`on_retriever_end` callbacks (most built-in retrievers do).
- If `node_run` spans look like generic chains rather than named nodes, the graph may not use LangGraph's standard node serialization. The tracer falls back to `serialized.get("name")`.

## Package layout

Key entry points in this repo:

```text
src/agentstep/sdk/tracer.py    replay_trace(), ReplayCallbackHandler
src/agentstep/sdk/exporter.py  ReplayOtelExporter (OTel → SQLite)
src/agentstep/server/cli.py    replay-debugger command
src/agentstep/server/api.py    FastAPI routes
src/agentstep/server/replayer.py branch replay logic
sample.py                      local demo graph
```

## Trace storage format

All spans are stored in a single SQLite table `otel_spans`:

| Column | Type | Description |
|---|---|---|
| `span_id` | TEXT (PK) | OpenTelemetry span context ID |
| `trace_id` | TEXT | OpenTelemetry trace context ID |
| `parent_span_id` | TEXT | Parent's span ID, or NULL for root spans |
| `name` | TEXT | Span type: one of `llm_call`, `chat_model_call`, `retriever_call`, `node_run`, `agent_step`, `tool_call` |
| `start_time` | INTEGER | Nanosecond timestamp |
| `end_time` | INTEGER | Nanosecond timestamp (NULL if span not yet ended) |
| `attributes` | TEXT | JSON-serialized attribute dictionary |
| `events` | TEXT | JSON array of OpenTelemetry events on the span |
| `status_code` | TEXT | `"OK"`, `"ERROR"`, or `"UNSET"` |
| `thread_id` | TEXT | Extracted for fast indexing |

The table includes an index on `thread_id` for efficient per-thread queries.