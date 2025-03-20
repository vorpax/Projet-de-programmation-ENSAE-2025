"""
Unit tests for comparing solver performances against known best scores.

This module tests different solver algorithms against the best scores
recorded in best-score.txt to ensure they achieve optimal or near-optimal results.
"""

import sys
import unittest
import re

sys.path.append("code/")
from grid import Grid
from solver import Solver, SolverGreedy, SolverFulkerson, SolverHungarian


class TestSolverBestScores(unittest.TestCase):
    """
    Test solver implementations against known best scores for different grid configurations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the best scores from the best-score.txt file once for all tests.
        """
        cls.best_scores = {}

        # Parse the best-score.txt file
        with open("tests/best-score.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_grid = None
        for line in lines:
            line = line.strip()
            if line == "---":
                continue
            elif line.endswith(".in"):
                current_grid = line
            elif current_grid and line.isdigit():
                cls.best_scores[current_grid] = int(line)

    def test_available_grids(self):
        """
        Verify that the best-score.txt file contains entries for all test grids.
        """
        # Check if the best-score.txt has entries
        self.assertGreater(
            len(self.best_scores), 0, "No best scores found in best-score.txt"
        )

        # Print all available grids for debug purposes
        print(f"Found {len(self.best_scores)} grid entries in best-score.txt:")
        for grid_name, best_score in self.best_scores.items():
            print(f"  {grid_name}: {best_score}")

    def _test_solver_on_grid(self, solver_class, grid_name, tolerance=0.2):
        """
        Helper method to test a solver on a specific grid.

        Parameters:
        -----------
        solver_class: class
            The solver class to test
        grid_name: str
            The name of the grid file to use
        tolerance: float
            The acceptable percentage deviation from the best score.
            Positive tolerance allows scores above best_score.
            Negative tolerance allows scores below best_score (which is good!)

        Returns:
        --------
        tuple
            (actual_score, best_score, is_within_tolerance)
        """
        # Find best score for this grid
        if grid_name not in self.best_scores:
            self.fail(f"No best score found for {grid_name}")

        best_score = self.best_scores[grid_name]

        # Load the grid
        try:
            grid = Grid.grid_from_file(f"input/{grid_name}", read_values=True)
            grid.cell_init()

            # Run the solver
            solver = solver_class(grid)
            solver.run()
            actual_score = solver.score()

            # Calculate deviation
            if best_score == 0:
                # Avoid division by zero
                deviation_percentage = 0 if actual_score == 0 else float("inf")
            else:
                deviation_percentage = ((actual_score - best_score) / best_score) * 100

            # For debugging
            print(
                f"{solver_class.__name__} on {grid_name}: Score={actual_score}, Best={best_score}, Deviation={deviation_percentage:.1f}%"
            )

            # Check if within tolerance:
            # - Scores above best_score should be within positive tolerance
            # - Scores below best_score are always acceptable (that's better than expected!)
            if actual_score < best_score:
                is_within_tolerance = True
            else:
                # Calculate tolerance range for scores above best
                max_acceptable = best_score * (1 + tolerance)
                is_within_tolerance = actual_score <= max_acceptable

            return (actual_score, best_score, is_within_tolerance)

        except Exception as e:
            self.fail(f"Error testing {solver_class.__name__} on {grid_name}: {str(e)}")
            return (None, best_score, False)

    def test_hungarian_solver_small_grids(self):
        """
        Test the Hungarian solver on small grid files (grid0x.in).
        """
        # Find all small grid files in the best scores dictionary
        small_grids = [
            name
            for name in self.best_scores.keys()
            if re.match(r"grid0[0-5]\.in", name)
        ]

        # Test each small grid
        for grid_name in small_grids:
            actual_score, best_score, is_within_tolerance = self._test_solver_on_grid(
                SolverHungarian, grid_name
            )

            self.assertTrue(
                is_within_tolerance,
                f"SolverHungarian score for {grid_name} is {actual_score}, "
                f"which exceeds the best score {best_score}",
            )

    def test_fulkerson_solver_small_grids(self):
        """
        Test the Fulkerson solver on small grid files (grid0x.in).
        """
        # Find all small grid files in the best scores dictionary
        small_grids = [
            name
            for name in self.best_scores.keys()
            if re.match(r"grid0[0-5]\.in", name)
        ]

        # Test each small grid
        for grid_name in small_grids:
            actual_score, best_score, is_within_tolerance = self._test_solver_on_grid(
                SolverFulkerson, grid_name
            )

            self.assertTrue(
                is_within_tolerance,
                f"SolverFulkerson score for {grid_name} is {actual_score}, "
                f"which exceeds the best score {best_score}",
            )

    def test_greedy_solver_small_grids(self):
        """
        Test the Greedy solver on small grid files (grid0x.in).
        This test allows for higher tolerance since the greedy approach may not find optimal solutions.
        """
        # Find all small grid files in the best scores dictionary
        small_grids = [
            name
            for name in self.best_scores.keys()
            if re.match(r"grid0[0-5]\.in", name)
        ]

        # Test each small grid with a higher tolerance
        higher_tolerance = 0.3  # Allow 30% deviation for greedy solver
        for grid_name in small_grids:
            actual_score, best_score, is_within_tolerance = self._test_solver_on_grid(
                SolverGreedy, grid_name, tolerance=higher_tolerance
            )

            # For greedy, we just print a warning if it doesn't meet the tolerance
            if not is_within_tolerance:
                print(
                    f"WARNING: SolverGreedy score for {grid_name} is {actual_score}, "
                    f"which exceeds the best score {best_score} by more than {higher_tolerance*100}%"
                )

    def test_hungarian_solver_medium_grids(self):
        """
        Test the Hungarian solver on medium grid files (grid1x.in).
        """
        # Find all medium grid files in the best scores dictionary
        medium_grids = [
            name
            for name in self.best_scores.keys()
            if re.match(r"grid1[0-9]\.in", name)
        ]

        # Test each medium grid
        for grid_name in medium_grids:
            actual_score, best_score, is_within_tolerance = self._test_solver_on_grid(
                SolverHungarian, grid_name
            )

            self.assertTrue(
                is_within_tolerance,
                f"SolverHungarian score for {grid_name} is {actual_score}, "
                f"which exceeds the best score {best_score}",
            )

    def test_specific_grid_with_all_solvers(self):
        """
        Test a specific grid with all available solvers and compare their performances.
        """
        grid_name = "grid01.in"  # Choose a simple grid for this test

        # Test with each solver
        solvers = [SolverGreedy, SolverFulkerson, SolverHungarian]
        results = {}

        for solver_class in solvers:
            actual_score, best_score, _ = self._test_solver_on_grid(
                solver_class, grid_name
            )
            results[solver_class.__name__] = actual_score

        # Print comparison
        print(f"\nSolver comparison for {grid_name} (Best score: {best_score}):")
        for solver_name, score in results.items():
            print(f"  {solver_name}: {score}")

        # Find the solver with the best (lowest) score
        best_solver = min(results.items(), key=lambda x: x[1])
        print(f"  Best solver: {best_solver[0]} with score {best_solver[1]}")


if __name__ == "__main__":
    unittest.main()
