"""
Unit tests for the Solver classes.
"""

import sys
import unittest


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

        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

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

        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

        self.solver = SolverEmpty(self.grid)

    def test_run(self):
        """
        Test run method.
        """
        result = self.solver.run()
        self.assertEqual(result, [])  # SolverEmpty.run() renvoie rien (c'est triste)
        self.assertEqual(self.solver.pairs, [])  # Subuwu


class TestSolverGreedy(unittest.TestCase):
    """
    Test the SolverGreedy class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """

        self.grid = Grid(2, 3)
        self.grid.color = [[0, 0, 0], [0, 0, 0]]
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

        self.solver = SolverGreedy(self.grid)

    def test_run(self):
        """
        Test run method.
        """

        pairs = self.solver.run()

        self.assertGreater(len(pairs), 0)

        self.assertEqual(self.solver.pairs, pairs)

        self.assertGreater(len(self.solver.cells), 0)

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

        self.grid = Grid(2, 2)
        self.grid.color = [[0, 0], [0, 0]]
        self.grid.value = [[1, 2], [3, 4]]

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

    def test_ford_fulkerson(self):
        """
        Test Ford-Fulkerson algorithm.
        """
        max_flow = self.solver.ford_fulkerson()

        self.assertEqual(max_flow, 2)

    def test_run(self):
        """
        Test run method.
        """

        matching_pairs = self.solver.run()

        self.assertEqual(len(matching_pairs), 2)

        self.assertEqual(self.solver.pairs, matching_pairs)

        self.assertEqual(len(self.solver.cells), 4)  # 2 paires = 4 cellules


if __name__ == "__main__":
    unittest.main()
