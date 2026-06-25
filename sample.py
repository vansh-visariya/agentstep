import os
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_replay.sdk.tracer import replay_trace

load_dotenv()

USE_FAKE = os.environ.get("REPLAY_USE_FAKE_LLM", "").lower() in ("1", "true", "yes")

# ── State ──────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Tools ──────────────────────────────────────────────────────

@tool
def get_weather(location: str):
    """Call to get the current weather in a location."""
    if location.lower() in ("sf", "san francisco"):
        return "It's 60 degrees and foggy in San Francisco."
    return f"It's 75 degrees and sunny in {location}."

tools = [get_weather]
tool_node = ToolNode(tools)

# ── LLM ────────────────────────────────────────────────────────

class FakeWeatherLLM(BaseChatModel):
    """Fake LLM that tool-calls get_weather for weather queries,
    and responds directly otherwise. No API key needed."""

    def bind_tools(self, tools, **kwargs):
        """Fake bind_tools — stores tools but doesn't use them for generation."""
        self._tools = tools
        return self

    def _generate(self, messages, stop=None, **kwargs):
        last = messages[-1]
        # If the last message is a tool result, summarise and return
        if last.type == "tool":
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=f"The weather result: {last.content}"))
            ])
        # First user message — call the weather tool
        if "weather" in (last.content or "").lower():
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "get_weather",
                        "args": {"location": "San Francisco"},
                        "id": "fake_call_1",
                    }]
                ))
            ])
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content=f"You said: {last.content}"))
        ])

    @property
    def _llm_type(self) -> str:
        return "fake-weather"

if USE_FAKE or not os.environ.get("GROQ_API_KEY"):
    print("Using fake LLM (no API key needed)")
    llm = FakeWeatherLLM()
else:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    print("Using Groq LLM")

llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ── Graph ──────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {END: END, "tools": "tools"})
    workflow.add_edge("tools", "agent")

    conn = sqlite3.connect("trace.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)

graph = build_graph()

# ── Demo runner ────────────────────────────────────────────────

def run_sample():
    config = {"configurable": {"thread_id": "demo_thread_1"}}

    with replay_trace(config, sqlite_path="trace.sqlite"):
        inputs = {"messages": [("user", "What's the weather in SF?")]}
        print("\nRunning agent...\n")
        for chunk in graph.stream(inputs, config, stream_mode="values"):
            msg = chunk["messages"][-1]
            print(f"  [{msg.type}] {msg.content[:200] if msg.content else '(tool call)'}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    -> {tc['name']}({tc['args']})")
        print("\nDone. Trace written to trace.sqlite")

if __name__ == "__main__":
    run_sample()
