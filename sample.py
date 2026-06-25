import sqlite3
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_replay.sdk.tracer import replay_trace

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define Tools
@tool
def get_weather(location: str):
    """Call to get the current weather in a location."""
    if location.lower() == "sf" or "san francisco" in location.lower():
        return "It's 60 degrees and foggy in San Francisco."
    return f"It's 75 degrees and sunny in {location}."

tools = [get_weather]
tool_node = ToolNode(tools)

# 3. Define Mock Model Node
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.runnables import RunnableConfig

def call_model(state: State, config: RunnableConfig):
    # Mocking the model deciding to call the tool
    messages = state["messages"]
    last_message = messages[-1]
    
    from agent_replay.sdk.tracer import trace
    tracer = trace.get_tracer("agent-replay")
    thread_id = config.get("configurable", {}).get("thread_id")
    
    if last_message.type == "tool":
        # The tool was called, return final response
        content = f"The weather is: {last_message.content}"
        
        # Manually emit OTel span for final response
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("lg.thread_id", thread_id)
            span.set_attribute("gen_ai.system", "langgraph")
            span.set_attribute("gen_ai.prompt", "Tool output received")
            span.set_attribute("gen_ai.completion", content)
            
        return {"messages": [AIMessage(content=content)]}
    
    # Otherwise, call the tool
    tool_call = ToolCall(name="get_weather", args={"location": "San Francisco"}, id="call_123")
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    
    # Manually emit OTel span for tool call request
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("lg.thread_id", thread_id)
        span.set_attribute("gen_ai.system", "langgraph")
        span.set_attribute("gen_ai.prompt", last_message.content)
        span.set_attribute("gen_ai.completion", "Tool called")
        
    return {"messages": [ai_msg]}

# 4. Define Graph Nodes
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 5. Build Graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile with SQLite checkpointer
conn = sqlite3.connect("trace.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)

def run_sample():
    # Run a test execution
    config = {"configurable": {"thread_id": "test_thread_1"}}
    
    with replay_trace(config, sqlite_path="trace.sqlite"):
        inputs = {"messages": [("user", "What's the weather in SF?")]}
        print("Running agent...")
        for chunk in graph.stream(inputs, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    run_sample()
