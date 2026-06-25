from typing import Any
from langchain_core.messages import BaseMessage, AnyMessage

def replay_branch(graph, config: dict, node_name: str, new_values: dict | Any) -> dict:
    """
    Forks execution from a specific checkpoint with modified values.
    
    Args:
        graph: The compiled LangGraph.
        config: The config containing thread_id and checkpoint_id to fork from.
        node_name: The name of the node where the change is injected (e.g., the tool node).
        new_values: The updated state values (e.g., {"messages": [ToolMessage(...)]}).
        
    Returns:
        The state after re-running the graph from the branch point.
    """
    # 1. Update the state at the specific checkpoint.
    # This creates a new branch in LangGraph's checkpoint history.
    new_config = graph.update_state(
        config=config,
        values=new_values,
        as_node=node_name
    )
    
    # 2. Resume execution from the new branched state.
    # Passing None as input means we just continue from the current state.
    final_state = graph.invoke(None, config=new_config)
    
    return final_state
