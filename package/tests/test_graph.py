# Test for Graph Functionality

import pytest
import networkx as nx
from src.core.graph import build_graph

def test_build_graph():
    # Sample data for testing
    data = {
        'jobID': ['Epigenomes(124)', 'Epigenomes(124)'],
        'taskID': [1, 2],
        'CPU': [0.2, 0.3],
        'RAM': [0.1, 0.2],
        'disk': [0.1, 0.1],
        'parent_task': [0, 1],
        'Runtime_C1': [10, 20],
        'Runtime_C2': [15, 25],
        'Runtime_C3': [5, 10],
        'deadline': [100, 100],
        'task_type': ['type1', 'type2']
    }

    # Build the graph
    G = build_graph(data)

    # Check if the graph is created correctly
    assert isinstance(G, nx.DiGraph), "Graph should be a directed graph"
    assert len(G.nodes) == 2, "Graph should have 2 nodes"
    assert len(G.edges) == 1, "Graph should have 1 edge"
    assert G.nodes[1]['CPU'] == 0.2, "Node 1 should have correct CPU value"
    assert G.nodes[2]['parent_task'] == 1, "Node 2 should have correct parent task"

def test_graph_properties():
    # Create a sample graph
    G = nx.DiGraph()
    G.add_node(1, CPU=0.2, RAM=0.1)
    G.add_node(2, CPU=0.3, RAM=0.2)
    G.add_edge(1, 2)

    # Check properties
    assert G.number_of_nodes() == 2, "Graph should have 2 nodes"
    assert G.number_of_edges() == 1, "Graph should have 1 edge"
    assert G.has_edge(1, 2), "Graph should have an edge from node 1 to node 2"