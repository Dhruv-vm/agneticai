"""Minimal smoke tests for verifying the Phase 0 research environment."""

import sys

import networkx as nx
import pydantic


def test_python_version() -> None:
    """Verify the environment runs Python 3.12."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_core_dependencies_importable() -> None:
    """Verify core graph and validation dependencies are available."""
    graph = nx.DiGraph()
    graph.add_edge("denial_node", "subsequent_call", relationship="COUNTERFACTUAL")
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert pydantic.__version__.startswith("2.")


def test_project_packages_importable() -> None:
    """Verify all Phase 0 src subpackages can be imported."""
    import src.agent
    import src.evaluation
    import src.monitor
    import src.replay
    import src.tools

    assert all(
        mod is not None
        for mod in (src.agent, src.evaluation, src.monitor, src.replay, src.tools)
    )
