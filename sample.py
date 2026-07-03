"""AgentStep demo — exercises all 6 span types.

Run without API keys:    uv run sample.py
Then launch the UI:      replay-debugger trace.sqlite --app sample:graph

Span types demonstrated:
  llm_call          — legacy LLM inference (on_llm_start/end)
  chat_model_call   — modern chat model inference (on_chat_model_start/end)
  retriever_call    — RAG retrieval queries (on_retriever_start/end)
  node_run          — graph node execution (on_chain_start/end)
  agent_step        — multi-step reasoning decisions (on_agent_action/finish)
  tool_call         — tool invocations (on_tool_start/end)
"""

import os
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.retrievers import BaseRetriever

from agentstep.sdk.tracer import replay_trace


# ── Fake LLM (no API keys needed) ───────────────────────────────

class FakeWeatherLLM(BaseChatModel):
    def __init__(self):
        super().__init__()
        self._tools = []

    def bind_tools(self, tools, **kwargs):
        self._tools = tools
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1] if isinstance(messages, list) else messages
        content = getattr(last, "content", "") or ""

        if isinstance(last, ToolMessage):
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=f"The weather result: {content}"))
            ])

        if "weather" in content.lower():
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                        "id": "fake_call_1",
                    }],
                ))
            ])

        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content=f"You said: {content}"))
        ])

    @property
    def _llm_type(self) -> str:
        return "fake-weather"


# ── Tools ───────────────────────────────────────────────────────

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    if "sf" in location.lower() or "san francisco" in location.lower():
        return "It's 60 degrees and foggy in San Francisco."
    return f"It's 75 degrees and sunny in {location}."


# ── Mock Retriever ──────────────────────────────────────────────

class MockRetriever(BaseRetriever):
    _docs: list[Document] = []

    def __init__(self):
        super().__init__()
        self._docs = [
            Document(
                page_content="LangChain is an open-source framework for building applications powered by large language models.",
                metadata={"source": "langchain_docs", "page": 1},
            ),
            Document(
                page_content="Retrieval Augmented Generation (RAG) improves factual accuracy by grounding LLM responses in external knowledge.",
                metadata={"source": "research_paper", "page": 3, "authors": ["Smith et al."]},
            ),
        ]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self._docs


# ── State ───────────────────────────────────────────────────────

class WeatherState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]


class RagState(TypedDict):
    question: str
    context: str
    answer: str


# ── Graph builders ──────────────────────────────────────────────

def build_weather_graph():
    """Triggers: llm_call, chat_model_call, tool_call, node_run."""
    fake_llm = FakeWeatherLLM()
    llm_with_tools = fake_llm.bind_tools([get_weather])

    def call_model(state):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(WeatherState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([get_weather]))
    workflow.add_edge(START, "agent")

    def should_call_tool(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            tc = last.tool_calls[-1]
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name == "get_weather":
                return "tools"
        return END

    workflow.add_conditional_edges("agent", should_call_tool, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(conn))


def build_rag_graph():
    """Triggers: retriever_call, llm_call, node_run."""
    retriever = MockRetriever()
    rag_llm = FakeWeatherLLM()

    def retrieve(state):
        # BaseRetriever.invoke() dispatches on_retriever_start/_end when the
        # LangChain callback manager is on the call path.  Calling the
        # internal method directly skips the manager, so always use invoke().
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(d.page_content for d in docs)
        return {"context": context}

    def synthesize(state):
        from langchain_core.messages import HumanMessage
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{state['context']}\n\n"
            f"Question: {state['question']}\nAnswer:"
        )
        response = rag_llm.invoke([HumanMessage(content=prompt)])
        return {"answer": getattr(response, "content", str(response))}

    workflow = StateGraph(RagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("synthesize", synthesize)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", END)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(conn))


def build_multi_step_graph():
    """Triggers: agent_step, tool_call, llm_call."""
    fake_llm = FakeWeatherLLM()
    llm_with_tools = fake_llm.bind_tools([get_weather])

    def call_model(state):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(WeatherState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([get_weather]))
    workflow.add_edge(START, "agent")

    def should_call_tool(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            tc = last.tool_calls[-1]
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name == "get_weather":
                return "tools"
        return END

    workflow.add_conditional_edges("agent", should_call_tool, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    return workflow.compile(checkpointer=SqliteSaver(conn))


# ── Runner ──────────────────────────────────────────────────────

# Reusable global graph for the UI's "load graph" call (--app sample:graph).
graph = None


def _get_graph():
    """Lazy singleton so `replay-debugger --app sample:graph` can load the graph."""
    global graph
    if graph is None:
        graph = build_weather_graph()
    return graph


SCENARIOS = [
    ("weather", "What's the weather in San Francisco?", build_weather_graph),
    ("rag", "What is retrieval augmented generation?", build_rag_graph),
    ("multi_step", "Plan a trip: check weather in Tokyo and Paris.", build_multi_step_graph),
]


def run_scenario(name, user_input, graph_builder):
    thread_id = f"demo_{name}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    print(f"\n[{thread_id}] starting", flush=True)

    g = graph_builder()

    initial_state = {"question": user_input} if name == "rag" else {"messages": [user_input]}

    with replay_trace(config, sqlite_path="trace.sqlite"):
        result = g.invoke(initial_state, config=config)

    print(f"[{thread_id}] done", flush=True)
    if isinstance(result, dict):
        for msg in result.get("messages", []):
            content = getattr(msg, "content", "") or ""
            print(f"  [{getattr(msg, 'type', 'unknown')}] {content[:200]}")
        if result.get("answer"):
            print(f"  [answer] {result['answer'][:200]}")

    try:
        conn = sqlite3.connect("trace.sqlite", timeout=5.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, COUNT(*) FROM otel_spans WHERE thread_id = ? GROUP BY name ORDER BY name",
            (thread_id,),
        )
        print(f"[{thread_id}] spans:")
        for span_type, count in cursor.fetchall():
            print(f"    {span_type}: {count}")
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"[{thread_id}] (could not read spans: {e})")
    return thread_id


def main():
    # Reset the trace tables in-place.  Removing the SQLite file on Windows
    # while a prior run still holds the handle leaves a phantom — new
    # connections create a fresh file, so the exporter's writes are lost.
    for f in ("trace.sqlite", "checkpoints.db"):
        try:
            conn = sqlite3.connect(f, timeout=5.0)
            conn.execute("DELETE FROM otel_spans")
            conn.execute("DELETE FROM checkpoints")
            conn.execute("DELETE FROM writes")
            conn.execute("DELETE FROM pending_writes")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass  # table doesn't exist yet on first run

    # SqliteSaver requires its tables to be created before first use.
    saver_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    SqliteSaver(saver_conn).setup()
    saver_conn.close()

    print("AgentStep — running all scenarios\n", flush=True)
    thread_ids = []
    for name, user_input, builder in SCENARIOS:
        thread_ids.append(run_scenario(name, user_input, builder))

    print("\n" + "=" * 60)
    print("All scenarios done. Launch the UI:")
    print("  replay-debugger trace.sqlite --app sample:graph")
    print("Thread IDs:")
    for tid in thread_ids:
        print(f"  {tid}")


if __name__ == "__main__":
    main()
