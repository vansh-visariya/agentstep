"""Tests for agentstep tracer — span capture and branch replay.

Run with: python -m tests.test_tracer
Or individually: python -m tests.test_tracer test_span_types_captured
"""

import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from typing import Annotated, TypedDict

# Ensure the package is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agentstep.sdk.tracer import replay_trace, ReplayCallbackHandler


# ── Test helpers ───────────────────────────────────────────────

PASSED = 0
FAILED = 0


def _check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)


def _read_spans(db_path: str, thread_id: str):
    """Load all spans for a thread from the SQLite trace file."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, attributes FROM otel_spans WHERE thread_id = ? ORDER BY start_time",
            (thread_id,),
        )
        spans = []
        for name, attrs_json in cursor.fetchall():
            spans.append({"name": name, "attributes": json.loads(attrs_json)})
    finally:
        conn.close()
    return spans


def _safe_unlink(path: str):
    """Delete a file, silently ignoring permission errors (Windows file locks)."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _assert_span_type(spans, span_name: str):
    """Find the first span of a given type."""
    for s in spans:
        if s["name"] == span_name:
            return s["attributes"]
    return None


# ── Fake LLM (same as sample.py) ─────────────────────────────

class FakeWeatherLLM:
    def __init__(self):
        self._tools = []

    def bind_tools(self, tools, **kwargs):
        self._tools = tools
        return self

    def invoke(self, messages, **kwargs):
        last = messages[-1] if isinstance(messages, list) else messages
        content = getattr(last, "content", "") or ""
        msg_type = getattr(last, "type", "")

        if msg_type == "tool":
            return AIMessage(content=f"The weather result: {content}")

        if "weather" in content.lower():
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                    "id": "fake_call_1",
                }],
            )

        return AIMessage(content=f"You said: {content}")


# ── Test 1: All span types captured ───────────────────────────

def test_span_types_captured():
    """Run a LangGraph agent and verify all 6 span types appear in the trace."""
    print("\n[test_span_types_captured]")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        # Build graph.
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @tool
        def get_weather(location: str):
            """Get the weather."""
            return "It's 60 degrees and foggy in San Francisco."

        tools = [get_weather]
        tool_node = ToolNode(tools)

        llm = FakeWeatherLLM()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
        workflow.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        graph = workflow.compile(checkpointer=__import__("langgraph.checkpoint.sqlite").checkpoint.sqlite.SqliteSaver(conn))

        # Run with tracing.
        config = {"configurable": {"thread_id": "test_thread_1"}}
        with replay_trace(config, sqlite_path=db_path):
            graph.invoke({"messages": [("user", "What's the weather in SF?")]}, config)

        conn.close()  # Close before reading spans to release file lock on Windows.

        # Inspect spans.
        spans = _read_spans(db_path, "test_thread_1")

        _check("llm_call span exists", any(s["name"] == "llm_call" for s in spans))
        llm_span = _assert_span_type(spans, "llm_call")
        _check("llm_call has gen_ai.completion", bool(llm_span and llm_span.get("gen_ai.completion")))
        _check("llm_call has gen_ai.system", bool(llm_span and llm_span.get("gen_ai.system")))

        _check("tool_call span exists", any(s["name"] == "tool_call" for s in spans))
        tool_span = _assert_span_type(spans, "tool_call")
        _check("tool_call has gen_ai.tool.name", bool(tool_span and tool_span.get("gen_ai.tool.name")))
        _check("tool_call has gen_ai.tool.output", bool(tool_span and tool_span.get("gen_ai.tool.output")))

        # node_run and agent_step may or may not appear depending on the LangGraph version.
        # We check for their presence as optional but verify they have correct attributes when present.
        node_spans = [s for s in spans if s["name"] == "node_run"]
        _check("node_run span exists (optional)", len(node_spans) > 0, f"found {len(node_spans)}")
        if node_spans:
            attrs = node_spans[0]["attributes"]
            _check("node_run has gen_ai.node.name", bool(attrs.get("gen_ai.node.name")))

        agent_spans = [s for s in spans if s["name"] == "agent_step"]
        _check("agent_step span exists (optional)", len(agent_spans) > 0, f"found {len(agent_spans)}")
        if agent_spans:
            attrs = agent_spans[0]["attributes"]
            _check("agent_step has gen_ai.agent.tool", bool(attrs.get("gen_ai.agent.tool")))

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass  # Windows may hold the file lock briefly


# ── Test 2: Retriever span captured (via mock callback) ───────

def test_retriever_span_captured():
    """Simulate a retriever call and verify the span is recorded."""
    print("\n[test_retriever_span_captured]")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        from agentstep.sdk.exporter import ReplayOtelExporter
        conn = sqlite3.connect(db_path, check_same_thread=False)
        exporter = ReplayOtelExporter(conn)

        handler = ReplayCallbackHandler(thread_id="retriever_test")

        # Simulate retriever start.
        handler.on_retriever_start(
            serialized={"id": "test_retriever.v1"},
            query="what is langchain?",
            run_id="run_ret_start",
        )

        # Simulate retriever end with mock documents.
        class FakeDoc:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = metadata or {}

        docs = [
            FakeDoc("LangChain is a framework for LLM applications.", {"source": "wiki", "page": 1}),
            FakeDoc("Retrieval augmented generation improves factual accuracy.", {"source": "paper", "page": 3}),
        ]
        handler.on_retriever_end(documents=docs, run_id="run_ret_start")

        # Flush spans to DB.
        from opentelemetry import trace as otel_trace
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

        conn.commit()
        conn.close()

        spans = _read_spans(db_path, "retriever_test")
        ret_span = _assert_span_type(spans, "retriever_call")

        _check("retriever_call span exists", bool(ret_span))
        if ret_span:
            _check("retriever has gen_ai.query", ret_span.get("gen_ai.query") == "what is langchain?")
            _check("retriever has document_count=2", ret_span.get("gen_ai.retriever.document_count") == 2)
            _check(
                "retriever first_doc_preview truncated to 500",
                len(ret_span.get("gen_ai.retriever.first_doc_preview", "")) <= 500,
            )
            meta_keys = json.loads(ret_span.get("gen_ai.retriever.metadata_keys", "[]"))
            _check("retriever metadata_keys has 'source'", "source" in meta_keys)

    finally:
        os.unlink(db_path)


# ── Test 3: Branch from tool_call works ───────────────────────

def test_branch_from_tool_call():
    """Create a trace, then branch from the tool_call span with overridden output."""
    print("\n[test_branch_from_tool_call]")
    db_path = tempfile.mktemp(suffix=".sqlite")

    try:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return "It's 60 degrees and foggy in San Francisco."

        tools = [get_weather]
        tool_node = ToolNode(tools)
        llm = FakeWeatherLLM()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
        workflow.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # ponytail: allow concurrent reads while graph holds write lock
        graph = workflow.compile(checkpointer=__import__("langgraph.checkpoint.sqlite").checkpoint.sqlite.SqliteSaver(conn))

        # Run original.
        config = {"configurable": {"thread_id": "branch_test"}}
        with replay_trace(config, sqlite_path=db_path):
            graph.invoke({"messages": [("user", "What's the weather in SF?")]}, config)

        # DON'T close conn — checkpointer needs it alive for state_history.

        # Find checkpoint where tools node is next.
        checkpoints = list(graph.get_state_history({"configurable": {"thread_id": "branch_test"}}))
        original_checkpoint = None
        for cp in reversed(checkpoints):
            if cp.next and "tools" in cp.next:
                original_checkpoint = cp
                break

        _check("found checkpoint with tools as next node", original_checkpoint is not None)
        if original_checkpoint is None:
            return

        # Branch from the tool_call span — replicate what /api/branch does internally.
        import uuid

        branch_id = f"branch_{uuid.uuid4().hex[:12]}"
        overridden_msg = ToolMessage(
            content="It's snowing in SF at 20°F.",
            tool_call_id="fake_call_1",
        )
        new_config = graph.update_state(
            config=original_checkpoint.config,
            values={"messages": [overridden_msg]},
            as_node="tools",
        )
        new_config.setdefault("configurable", {})
        if "checkpoint_ns" not in new_config["configurable"]:
            new_config["configurable"]["checkpoint_ns"] = original_checkpoint.config.get("configurable", {}).get("checkpoint_ns", "")

        # Run with tracing and branch_id.
        branched_config = {"configurable": {
            "thread_id": "branch_test",
            "checkpoint_id": new_config["configurable"]["checkpoint_id"],
        }}
        with replay_trace(branched_config, sqlite_path=db_path, branch_id=branch_id):
            graph.invoke(None, config=new_config)

        # Verify the branched trace has a new branch.
        spans = _read_spans(db_path, "branch_test")
        branches = {}
        for s in spans:
            bid = s["attributes"].get("lg.branch_id") or "__original__"
            branches.setdefault(bid, []).append(s)

        branch_span_count = len(branches.get(branch_id, []))
        _check("branch has new spans", branch_span_count > 0, f"found {branch_span_count} spans in branch")

        # The overridden tool output should appear.
        branched_tool_spans = [s for s in branches.get(branch_id, []) if s["name"] == "tool_call"]
        _check("branched tool has new output", any(
            "snowing" in (s["attributes"].get("gen_ai.tool.output") or "").lower()
            for s in branched_tool_spans
        ), f"tool outputs: {[s['attributes'].get('gen_ai.tool.output', '') for s in branched_tool_spans]}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            os.unlink(db_path)
        except OSError:
            pass  # Windows may hold the file lock briefly


# ── Test 4: Branch from llm_call works ────────────────────────

def test_branch_from_llm_call():
    """Branch from an LLM call with overridden completion."""
    print("\n[test_branch_from_llm_call]")
    db_path = tempfile.mktemp(suffix=".sqlite")

    try:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return "It's 60 degrees and foggy in San Francisco."

        tools = [get_weather]
        tool_node = ToolNode(tools)
        llm = FakeWeatherLLM()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
        workflow.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        graph = workflow.compile(checkpointer=__import__("langgraph.checkpoint.sqlite").checkpoint.sqlite.SqliteSaver(conn))

        config = {"configurable": {"thread_id": "llm_branch_test"}}
        with replay_trace(config, sqlite_path=db_path):
            graph.invoke({"messages": [("user", "What's the weather in SF?")]}, config)

        # DON'T close conn — checkpointer needs it alive.

        # Find checkpoint where agent is next.
        checkpoints = list(graph.get_state_history({"configurable": {"thread_id": "llm_branch_test"}}))
        llm_checkpoint = None
        for cp in reversed(checkpoints):
            if cp.next and "agent" in cp.next:
                llm_checkpoint = cp
                break

        _check("found checkpoint with agent as next node", llm_checkpoint is not None)
        if llm_checkpoint is None:
            return

        # Branch from the llm_call span.
        import uuid
        branch_id = f"branch_{uuid.uuid4().hex[:12]}"
        overridden_msg = AIMessage(content="The weather is sunny and 72°F.")
        new_config = graph.update_state(
            config=llm_checkpoint.config,
            values={"messages": [overridden_msg]},
            as_node="agent",
        )
        new_config.setdefault("configurable", {})
        if "checkpoint_ns" not in new_config["configurable"]:
            new_config["configurable"]["checkpoint_ns"] = llm_checkpoint.config.get("configurable", {}).get("checkpoint_ns", "")

        branched_config = {"configurable": {
            "thread_id": "llm_branch_test",
            "checkpoint_id": new_config["configurable"]["checkpoint_id"],
        }}
        with replay_trace(branched_config, sqlite_path=db_path, branch_id=branch_id):
            graph.invoke(None, config=new_config)

        spans = _read_spans(db_path, "llm_branch_test")
        branches = {}
        for s in spans:
            bid = s["attributes"].get("lg.branch_id") or "__original__"
            branches.setdefault(bid, []).append(s)

        branched_llm = [s for s in branches.get(branch_id, []) if s["name"] == "llm_call"]
        _check("branched LLM has new completion", any(
            "sunny" in (s["attributes"].get("gen_ai.completion") or "").lower()
            for s in branched_llm
        ), f"completions: {[s['attributes'].get('gen_ai.completion', '')[:40] for s in branched_llm]}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            os.unlink(db_path)
        except OSError:
            pass  # Windows may hold the file lock briefly


# ── Test 5: Span attributes carry thread_id and branch_id ─────

def test_span_metadata():
    """Verify every span carries lg.thread_id and that branched spans have lg.branch_id."""
    print("\n[test_span_metadata]")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return "Sunny in SF."

        tools = [get_weather]
        tool_node = ToolNode(tools)
        llm = FakeWeatherLLM()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
        workflow.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        graph = workflow.compile(checkpointer=__import__("langgraph.checkpoint.sqlite").checkpoint.sqlite.SqliteSaver(conn))

        config = {"configurable": {"thread_id": "meta_test"}}
        with replay_trace(config, sqlite_path=db_path):
            graph.invoke({"messages": [("user", "What's the weather?")]}, config)

        spans = _read_spans(db_path, "meta_test")

        all_have_thread_id = all(s["attributes"].get("lg.thread_id") == "meta_test" for s in spans)
        _check("all spans have lg.thread_id='meta_test'", all_have_thread_id)

        no_branch_on_original = all(
            "lg.branch_id" not in s["attributes"] or s["attributes"]["lg.branch_id"] is None
            for s in spans
        )
        _check("original spans have no lg.branch_id", no_branch_on_original)

    finally:
        try:
            conn.close()
        except Exception:
            pass
        _safe_unlink(db_path)


# ── Test 6: Node run span has correct attributes ──────────────

def test_node_run_attributes():
    """Verify node_run spans capture node name and input/output keys."""
    print("\n[test_node_run_attributes]")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return "Sunny in SF."

        tools = [get_weather]
        tool_node = ToolNode(tools)
        llm = FakeWeatherLLM()
        llm_with_tools = llm.bind_tools(tools)

        def call_model(state):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        workflow = StateGraph(State)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
        workflow.add_edge("tools", "agent")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        graph = workflow.compile(checkpointer=__import__("langgraph.checkpoint.sqlite").checkpoint.sqlite.SqliteSaver(conn))

        config = {"configurable": {"thread_id": "node_test"}}
        with replay_trace(config, sqlite_path=db_path):
            graph.invoke({"messages": [("user", "What's the weather?")]}, config)

        spans = _read_spans(db_path, "node_test")
        node_spans = [s for s in spans if s["name"] == "node_run"]

        _check("at least one node_run span exists", len(node_spans) > 0, f"found {len(node_spans)}")
        if not node_spans:
            return

        attrs = node_spans[0]["attributes"]
        _check("node_run has gen_ai.node.name", bool(attrs.get("gen_ai.node.name")), f"attrs keys: {list(attrs.keys())}")
        node_name = attrs.get("gen_ai.node.name") or ""
        _check(
            "node name is 'agent' or 'tools'",
            node_name in ("agent", "tools"),
            f"got '{node_name}'",
        )

    finally:
        try:
            conn.close()
        except Exception:
            pass
        _safe_unlink(db_path)


# ── Test 7: Chat model span captured (modern LangChain path) ──

def test_chat_model_span_captured():
    """Simulate a chat model callback and verify the span is recorded."""
    print("\n[test_chat_model_span_captured]")
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        from agentstep.sdk.exporter import ReplayOtelExporter
        conn = sqlite3.connect(db_path, check_same_thread=False)
        exporter = ReplayOtelExporter(conn)

        handler = ReplayCallbackHandler(thread_id="chat_test")

        # Simulate chat model start.
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        handler.on_chat_model_start(
            serialized={"id": "openai:gpt-4"},
            messages=[[FakeMessage("Hello, world!")]],
            run_id="run_chat_start",
        )

        # Simulate chat model end.
        from langchain_core.outputs import LLMResult
        response = LLMResult(generations=[
            [ChatGeneration(message=AIMessage(content="Hi there! How can I help?"))]
        ], llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 8}})

        handler.on_chat_model_end(response=response, run_id="run_chat_start")

        from opentelemetry import trace as otel_trace
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

        conn.commit()
        conn.close()

        spans = _read_spans(db_path, "chat_test")
        chat_span = _assert_span_type(spans, "chat_model_call")

        _check("chat_model_call span exists", bool(chat_span))
        if chat_span:
            _check("chat model has gen_ai.system=openai", chat_span.get("gen_ai.system") == "openai")
            _check(
                "chat model has first_message_preview",
                "Hello, world!" in (chat_span.get("gen_ai.chat.first_message_preview") or ""),
            )
            _check("chat model has message_count=1", chat_span.get("gen_ai.chat.message_count") == 1)
            _check(
                "chat model has completion text",
                chat_span.get("gen_ai.completion") == "Hi there! How can I help?",
            )
            _check("chat model has input_tokens=10", chat_span.get("gen_ai.usage.input_tokens") == 10)

    finally:
        os.unlink(db_path)


# ── Runner ─────────────────────────────────────────────────────

def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"\n{'='*60}")
    print(f"  agentstep tracer tests — {len(tests)} test functions")
    print(f"{'='*60}\n")

    for t in tests:
        try:
            t()
        except Exception as e:
            FAILED += 1
            import traceback
            print(f"\n  ERROR  {t.__name__}:\n{traceback.format_exc()}")

    print(f"\n{'='*60}")
    print(f"  Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")
    print(f"{'='*60}\n")

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
