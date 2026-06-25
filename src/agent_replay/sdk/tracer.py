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

from agent_replay.sdk.exporter import ReplayOtelExporter

tracer_provider = None

def setup_otel(sqlite_path: str = "trace.sqlite"):
    global tracer_provider
    if tracer_provider is not None:
        return
        
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    exporter = ReplayOtelExporter(conn)
    
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

class ReplayCallbackHandler(BaseCallbackHandler):
    """
    A LangChain callback handler that emits OpenTelemetry spans for LLMs and Tools,
    enriching them with the LangGraph thread_id.
    """
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.tracer = trace.get_tracer("agent-replay")
        self.spans = {}  # run_id -> Span

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        span = self.tracer.start_span("llm_call")
        span.set_attribute("lg.thread_id", self.thread_id)
        span.set_attribute("gen_ai.system", "langgraph")
        if prompts:
            span.set_attribute("gen_ai.prompt", prompts[0])
        self.spans[str(run_id)] = span

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        span = self.spans.get(str(run_id))
        if span:
            # Capture output if available
            if response.generations and response.generations[0]:
                span.set_attribute("gen_ai.completion", response.generations[0][0].text)
            
            # Capture token usage
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
        span.set_attribute("lg.thread_id", self.thread_id)
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

@contextmanager
def replay_trace(config: dict, sqlite_path: str = "trace.sqlite"):
    """
    Context manager to wrap LangGraph executions and inject the OTel callback handler.
    Example:
        config = {"configurable": {"thread_id": "123"}}
        with replay_trace(config):
            graph.invoke(input_data, config=config)
    """
    setup_otel(sqlite_path)
    
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    
    handler = ReplayCallbackHandler(thread_id)
    
    # Inject the handler into the config's callbacks
    callbacks = config.get("callbacks", [])
    if isinstance(callbacks, list):
        callbacks.append(handler)
    else:
        callbacks = [callbacks, handler]
    
    config["callbacks"] = callbacks
    
    yield config
