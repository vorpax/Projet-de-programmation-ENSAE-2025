"""
Unit tests for the Solver classes.
"""

import sys
import unittest
from unittest.mock import patch

sys.path.append("code/")
from grid import Grid
from solver import Solver, SolverEmpty, SolverGreedy, SolverFulkerson


class TestSolver(unittest.TestCase):
    """
    Test the base Solver class functionality.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a test grid
        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]  # All white
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

        # Initialize solver
        self.solver = Solver(self.grid)

    def test_init(self):
        """
        Test initialization of solver.
        """
        self.assertEqual(self.solver.grid, self.grid)
        self.assertEqual(self.solver.pairs, [])
        self.assertEqual(self.solver.cells, [])
        self.assertEqual(
            self.solver.all_cells, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        )


class TestSolverEmpty(unittest.TestCase):
    """
    Test the SolverEmpty class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a test grid
        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]  # All white
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

        # Initialize solver
        self.solver = SolverEmpty(self.grid)

    def test_run(self):
        """
        Test run method.
        """
        result = self.solver.run()
        self.assertIsNone(result)  # SolverEmpty.run() doesn't return anything
        self.assertEqual(self.solver.pairs, [])  # No pairs should be chosen


class TestSolverGreedy(unittest.TestCase):
    """
    Test the SolverGreedy class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a test grid
        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]  # All white
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

        # Initialize solver
        self.solver = SolverGreedy(self.grid)

    @patch("builtins.print")  # Mock print to avoid output during tests
    def test_run(self, mock_print):
        """
        Test run method.
        """
        # Run the solver
        pairs = self.solver.run()

        # Verify that pairs were chosen
        self.assertGreater(len(pairs), 0)

        # Verify that pairs attribute is populated
        self.assertEqual(self.solver.pairs, pairs)

        # Verify that cells attribute is populated
        self.assertGreater(len(self.solver.cells), 0)

        # Check that each pair consists of valid cells
        for pair in pairs:
            self.assertEqual(len(pair), 2)
            for cell in pair:
                self.assertIn(cell, self.solver.all_cells)


class TestSolverFulkerson(unittest.TestCase):
    """
    Test the SolverFulkerson class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a small test grid
        self.grid = Grid(2, 2)
        self.grid.color = [[0, 0], [0, 0]]  # All white
        self.grid.value = [[1, 2], [3, 4]]

        # Initialize solver
        self.solver = SolverFulkerson(self.grid)

    def test_init(self):
        """
        Test initialization of Ford-Fulkerson solver.
        """
        self.assertIsInstance(self.solver.residual_graph, dict)
        self.assertIn("source", self.solver.residual_graph)
        self.assertIn("sink", self.solver.residual_graph)

        # Check cell nodes exist
        self.assertIn("cell_0_0", self.solver.residual_graph)
        self.assertIn("cell_0_1", self.solver.residual_graph)
        self.assertIn("cell_1_0", self.solver.residual_graph)
        self.assertIn("cell_1_1", self.solver.residual_graph)

    def test_adjacency_graph_init(self):
        """
        Test adjacency graph initialization.
        """
        # Even cells (0,0) and (1,1) should connect to source
        self.assertIn("cell_0_0", self.solver.residual_graph["source"])
        self.assertIn("cell_1_1", self.solver.residual_graph["source"])

        # Odd cells (0,1) and (1,0) should connect to sink
        self.assertIn("sink", self.solver.residual_graph["cell_0_1"])
        self.assertIn("sink", self.solver.residual_graph["cell_1_0"])

        # Check connections between even and odd cells
        self.assertIn(
            "cell_0_1", self.solver.residual_graph["cell_0_0"]
        )  # (0,0) -> (0,1)
        self.assertIn(
            "cell_1_0", self.solver.residual_graph["cell_0_0"]
        )  # (0,0) -> (1,0)
        self.assertIn(
            "cell_0_1", self.solver.residual_graph["cell_1_1"]
        )  # (1,1) -> (0,1)
        self.assertIn(
            "cell_1_0", self.solver.residual_graph["cell_1_1"]
        )  # (1,1) -> (1,0)

    def test_find_augmenting_path(self):
        """
        Test finding an augmenting path.
        """
        path = self.solver.find_augmenting_path()

        # A path should exist in this simple grid
        self.assertIsNotNone(path)

        # Path should start with source and end with sink
        self.assertEqual(path[0], "source")
        self.assertEqual(path[-1], "sink")

        # Path should alternate between nodes
        self.assertEqual(len(path), 4)  # source -> even cell -> odd cell -> sink

    @patch("builtins.print")  # Mock print to avoid output during tests
    def test_ford_fulkerson(self, mock_print):
        """
        Test Ford-Fulkerson algorithm.
        """
        max_flow = self.solver.ford_fulkerson()

        # Max flow should be 2 (two independent pairs possible in this grid)
        self.assertEqual(max_flow, 2)

    @patch("builtins.print")  # Mock print to avoid output during tests
    def test_run(self, mock_print):
        """
        Test run method.
        """
        # Run the solver
        matching_pairs = self.solver.run()

        # Verify that pairs were matched
        self.assertEqual(len(matching_pairs), 2)

        # Verify that pairs attribute is populated
        self.assertEqual(self.solver.pairs, matching_pairs)

        # Verify that cells attribute is populated
        self.assertEqual(len(self.solver.cells), 4)  # All cells are matched


if __name__ == "__main__":
    unittest.main()
