from typing import Any


def replay_branch(
    graph: Any,
    config: dict,
    node_name: str,
    new_values: dict | Any,
) -> dict:
    """Fork execution from a specific checkpoint with modified values.

    Args:
        graph: The compiled LangGraph with a checkpointer.
        config: Config containing ``thread_id`` and optionally ``checkpoint_id``.
        node_name: The node to associate the new values with (e.g. ``"tools"``).
        new_values: State update dict (e.g. ``{"messages": [ToolMessage(...)]}``).

    Returns:
        The final state after the branched execution completes.
    """
    # 1. Validate the graph has a checkpointer
    if not getattr(graph, "checkpointer", None):
        raise ValueError(
            "Graph must have a checkpointer to replay branches. "
            "Compile with e.g. checkpointer=SqliteSaver(conn)."
        )

    # 2. Validate the checkpoint exists
    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        raise ValueError(
            f"Could not load checkpoint for config {config}: {e}"
        ) from e

    # 3. Validate the node exists in the graph
    if node_name not in graph.nodes:
        raise ValueError(
            f"Node '{node_name}' not found in graph. "
            f"Available: {list(graph.nodes.keys())}"
        )

    # 4. Create a branched checkpoint with the overridden state
    new_config = graph.update_state(
        config=snapshot.config,
        values=new_values,
        as_node=node_name,
    )

    # Ensure checkpoint_ns is present — required by LangGraph for
    # resuming from a checkpoint via invoke() on some versions.
    new_config.setdefault("configurable", {})
    if "checkpoint_ns" not in new_config["configurable"]:
        new_config["configurable"]["checkpoint_ns"] = (
            snapshot.config.get("configurable", {}).get("checkpoint_ns", "")
        )

    # Propagate callbacks from the original config so the tracer
    # works during the branched execution.
    if "callbacks" in config:
        new_config["callbacks"] = config["callbacks"]

    # 5. Resume from the new branch
    try:
        return graph.invoke(None, config=new_config)
    except Exception as e:
        partial = graph.get_state(new_config)
        return {
            "status": "partial",
            "error": str(e),
            "checkpoint_id": partial.config["configurable"].get("checkpoint_id"),
        }
