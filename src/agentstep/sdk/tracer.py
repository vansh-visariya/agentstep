import functools
import sqlite3
import os
import json
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from agentstep.sdk.exporter import ReplayOtelExporter

# ponytail: module-level state for the singleton tracer provider.
# The exporter's connection is tracked so it can be shut down cleanly.
_tracer_provider = None
_exporter_conn = None


def setup_otel(sqlite_path: str = "trace.sqlite"):
    """Initialise the global OTel TracerProvider with a SQLite-backed exporter.

    Idempotent — if a provider already exists (from a previous replay_trace call),
    reuse it instead of creating a new one. OpenTelemetry doesn't allow overriding
    set_tracer_provider() once it's been called, so we keep the first provider alive
    across multiple replay_trace invocations in the same process.
    """
    global _tracer_provider, _exporter_conn

    # Reuse existing provider if it has an active exporter (from a previous run).
    if _tracer_provider is not None:
        return

    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    exporter = ReplayOtelExporter(conn)
    _exporter_conn = conn

    _tracer_provider = TracerProvider()
    _tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(_tracer_provider)


def teardown_otel():
    """Flush pending spans to the SQLite connection.

    Keeps the global TracerProvider and connection alive so subsequent
    ``replay_trace`` calls reuse them — OpenTelemetry doesn't allow overriding
    a once-set provider, and recreating the connection mid-process would lose
    in-flight spans.  The process exit closes the handle.
    """
    global _tracer_provider, _exporter_conn
    if _tracer_provider is not None:
        # SimpleSpanProcessor.on_end exports synchronously, so there is
        # nothing to flush.  We only need to commit any in-flight SQLite
        # writes from the exporter below — calling force_flush() on the
        # provider would shut the processor down and break subsequent runs.
        pass
    if _exporter_conn is not None:
        try:
            _exporter_conn.commit()
        except Exception:
            pass

class ReplayCallbackHandler(BaseCallbackHandler):
    """
    A LangChain callback handler that emits OpenTelemetry spans for LLMs and Tools,
    enriching them with the LangGraph thread_id and optional branch_id.
    """
    def __init__(self, thread_id: str, branch_id: str | None = None):
        self.thread_id = thread_id
        self.branch_id = branch_id
        self.tracer = trace.get_tracer("agentstep")
        self.spans = {}  # run_id -> Span
        self._chat_run_ids: set[str] = set()  # skip duplicate llm_* spans for chat models

    def _set_branch_attrs(self, span):
        span.set_attribute("lg.thread_id", self.thread_id)
        if self.branch_id:
            span.set_attribute("lg.branch_id", self.branch_id)

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("llm_call")
        self._set_branch_attrs(span)
        # Model id: prefer serialized["id"], fall back to class name in metadata.
        sys_id = (serialized or {}).get("id") or (metadata or {}).get("ls_provider", "langgraph")
        span.set_attribute("gen_ai.system", str(sys_id))
        if prompts:
            first = prompts[0]
            if isinstance(first, list):
                # Chat-style: prompts is List[List[BaseMessage]]
                try:
                    parts = []
                    for m in first:
                        c = getattr(m, "content", str(m))
                        if isinstance(c, list):
                            # content can be a list of content-parts (e.g. multimodal)
                            c = " ".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in c)
                        parts.append(f"{getattr(m, 'type', '?')}: {c}")
                    span.set_attribute("gen_ai.prompt", "\n".join(parts)[:2000])
                except Exception:
                    span.set_attribute("gen_ai.prompt", str(first)[:2000])
            else:
                span.set_attribute("gen_ai.prompt", str(first)[:2000])
        self.spans[str(run_id)] = span

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                # Chat models put the text in .message.content; legacy LLMs in .text.
                text = gen.text or getattr(getattr(gen, "message", None), "content", "") or ""
                if not text and getattr(gen, "message", None) is not None:
                    msg = gen.message
                    tc = getattr(msg, "tool_calls", None)
                    if tc:
                        text = f"[tool_call: {[t.get('name', getattr(t, 'name', '?')) for t in tc]}]"
                span.set_attribute("gen_ai.completion", str(text)[:2000])

            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                if "prompt_tokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])

            span.end()
            del self.spans[str(run_id)]

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("tool_call")
        self._set_branch_attrs(span)
        span.set_attribute("gen_ai.tool.name", serialized.get("name", "unknown_tool"))
        span.set_attribute("gen_ai.tool.input", input_str)
        self.spans[str(run_id)] = span

    def on_tool_end(self, output: str, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            span.set_attribute("gen_ai.tool.output", str(output))
            span.end()
            del self.spans[str(run_id)]

    def on_tool_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            span.record_exception(error)
            span.end()
            del self.spans[str(run_id)]

    # ── Retriever events (RAG retrieval tracing) ──────────────────────────

    def on_retriever_start(self, serialized: dict, query: str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("retriever_call")
        self._set_branch_attrs(span)
        # serialized can be None or missing 'id' in some LangGraph sub-chain invocations.
        if serialized:
            retriever_id = serialized.get("id", "unknown_retriever")
        else:
            retriever_id = "unknown_retriever"
        span.set_attribute("gen_ai.system", retriever_id)
        span.set_attribute("gen_ai.query", query)
        self.spans[str(run_id)] = span

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            span.set_attribute("gen_ai.retriever.document_count", len(documents))
            if documents:
                first_doc = documents[0]
                content_preview = getattr(first_doc, "page_content", "") or ""
                span.set_attribute("gen_ai.retriever.first_doc_preview", content_preview[:500])
                meta = getattr(first_doc, "metadata", {}) or {}
                if meta:
                    meta_keys = list(meta.keys())[:10]
                    span.set_attribute("gen_ai.retriever.metadata_keys", json.dumps(meta_keys))
            span.end()
            del self.spans[str(run_id)]

    def on_retriever_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            span.record_exception(error)
            span.end()
            del self.spans[str(run_id)]

    # ── Chain / node events (graph topology) ──────────────────────────────

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("node_run")
        self._set_branch_attrs(span)
        # serialized can be None or missing 'id' in some LangGraph sub-chain invocations.
        if serialized:
            chain_id = serialized.get("id", "")
            if chain_id and chain_id.startswith("langgraph:chain:"):
                node_name = chain_id.split("langgraph:chain:")[-1]
            else:
                node_name = serialized.get("name") or "unknown_node"
        else:
            node_name = "unknown_node"
        span.set_attribute("gen_ai.node.name", node_name)
        if inputs:
            input_keys = list(inputs.keys())[:20]
            span.set_attribute("gen_ai.node.input_keys", json.dumps(input_keys))
        self.spans[str(run_id)] = span

    def on_chain_end(self, outputs: dict | str, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            # outputs can be a string (e.g. single value from a chain) or a dict.
            if isinstance(outputs, dict):
                output_keys = list(outputs.keys())[:20]
                span.set_attribute("gen_ai.node.output_keys", json.dumps(output_keys))
            elif isinstance(outputs, str):
                span.set_attribute("gen_ai.node.output_preview", outputs[:200])
            span.end()
            del self.spans[str(run_id)]

    def on_chain_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            span.record_exception(error)
            span.end()
            del self.spans[str(run_id)]

    # ── Agent events (multi-step reasoning) ───────────────────────────────

    def on_agent_action(self, action, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("agent_step")
        self._set_branch_attrs(span)
        span.set_attribute("gen_ai.agent.tool", getattr(action, "tool", ""))
        tool_input = action.tool_input if hasattr(action, "tool_input") else None
        if isinstance(tool_input, dict):
            keys = list(tool_input.keys())[:10]
            span.set_attribute("gen_ai.agent.tool_input_keys", json.dumps(keys))
        elif tool_input is not None:
            span.set_attribute("gen_ai.agent.tool_input", str(tool_input)[:500])
        log = action.log if hasattr(action, "log") else ""
        if log:
            span.set_attribute("gen_ai.agent.log_preview", str(log)[:300])
        self.spans[str(run_id)] = span

    def on_agent_finish(self, finish, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            return_values = finish.return_values if hasattr(finish, "return_values") else {}
            if return_values:
                rv_keys = list(return_values.keys())[:10]
                span.set_attribute("gen_ai.agent.return_keys", json.dumps(rv_keys))
            log = getattr(finish, "log", "") or ""
            if log:
                span.set_attribute("gen_ai.agent.final_log_preview", str(log)[:300])
            span.end()
            del self.spans[str(run_id)]

    # ── Chat model events (modern LangChain chat models) ──────────────────

    def on_chat_model_start(self, serialized: dict, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._chat_run_ids.add(str(run_id))
        try:
            span = self.tracer.start_span("chat_model_call")
            self._set_branch_attrs(span)
            chat_id = (serialized or {}).get("id", "") if isinstance(serialized, dict) else ""
            if ":" in str(chat_id):
                span.set_attribute("gen_ai.system", str(chat_id).split(":")[0])
            else:
                span.set_attribute("gen_ai.system", chat_id or "unknown_chat_model")

            # Normalise `messages` — different LangChain versions pass it in
            # different shapes (List[List[BaseMessage]], List[BaseMessage],
            # or wrapped in a dict).  We extract a best-effort preview.
            norm = messages
            if isinstance(norm, dict):
                norm = norm.get("messages", norm.get("input", []))
            if isinstance(norm, list) and norm and not isinstance(norm[0], (list, tuple)) and hasattr(norm[0], "type"):
                # Flat list of messages — wrap to a single turn
                norm = [norm]
            preview = ""
            total_msgs = 0
            if isinstance(norm, list):
                for turn in norm:
                    if isinstance(turn, list):
                        total_msgs += len(turn)
                        if not preview and turn:
                            content = getattr(turn[0], "content", "") or ""
                            if isinstance(content, list):
                                content = " ".join(
                                    str(p.get("text", p)) if isinstance(p, dict) else str(p)
                                    for p in content
                                )
                            preview = str(content)
            span.set_attribute("gen_ai.chat.first_message_preview", preview[:500])
            span.set_attribute("gen_ai.chat.message_count", total_msgs)
            self.spans[str(run_id)] = span
        except Exception:
            # Malformed callback payload — drop this span, don't kill the trace.
            self._chat_run_ids.discard(str(run_id))

    def on_chat_model_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        self._chat_run_ids.discard(str(run_id))
        span = self.spans.get(str(run_id))
        if not span:
            return
        try:
            if response.generations and response.generations[0]:
                gen = response.generations[0][0]
                text = gen.text or getattr(getattr(gen, "message", None), "content", "") or ""
                if not text and getattr(gen, "message", None) is not None:
                    msg = gen.message
                    tc = getattr(msg, "tool_calls", None)
                    if tc:
                        text = f"[tool_call: {[t.get('name', getattr(t, 'name', '?')) for t in tc]}]"
                span.set_attribute("gen_ai.completion", str(text)[:2000])
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                if "prompt_tokens" in usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
                if "completion_tokens" in usage:
                    span.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
        except Exception:
            pass
        span.end()
        del self.spans[str(run_id)]

@contextmanager
def replay_trace(config: dict, sqlite_path: str = "trace.sqlite", branch_id: str | None = None):
    """
    Context manager to wrap LangGraph executions and inject the OTel callback handler.

    Example::

        config = {"configurable": {"thread_id": "123"}}
        with replay_trace(config):
            graph.invoke(input_data, config=config)

    On exit, flushes pending spans to disk and closes the exporter's SQLite
    connection so Windows file handles are released for subsequent readers.
    """
    setup_otel(sqlite_path)

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    handler = ReplayCallbackHandler(thread_id, branch_id=branch_id)

    # Inject the handler into the config's callbacks
    callbacks = config.get("callbacks", [])
    if isinstance(callbacks, list):
        callbacks.append(handler)
    else:
        callbacks = [callbacks, handler]

    config["callbacks"] = callbacks

    try:
        yield config
    finally:
        teardown_otel()
