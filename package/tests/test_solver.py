import unittest
from src.core.solver import solve_workflow_scheduling
from src.core.graph import build_graph
from src.core.loader import load_data
import networkx as nx

class TestSolver(unittest.TestCase):

    def setUp(self):
        # Load sample data for testing
        self.data = load_data('data/workflow.csv')
        self.graph = build_graph(self.data)

    def test_solve_workflow_scheduling(self):
        assignments, cost = solve_workflow_scheduling(self.graph)
        
        # Check if assignments are not empty
        self.assertIsNotNone(assignments)
        self.assertGreater(len(assignments), 0)

        # Check if the cost is a valid number
        self.assertIsInstance(cost, (int, float))

    def test_graph_structure(self):
        # Ensure the graph is a directed acyclic graph (DAG)
        self.assertTrue(nx.is_directed_acyclic_graph(self.graph))

    def test_task_assignments(self):
        assignments, _ = solve_workflow_scheduling(self.graph)
        
        # Check if all tasks are assigned to a processor
        for task in self.graph.nodes():
            self.assertIn(task, assignments)

if __name__ == '__main__':
    unittest.main()