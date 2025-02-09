"""
Unit test for the function grid_from_file in the Grid class.
"""

# This will work if ran from the root folder (the folder in which there is the subfolder code/)
import sys

sys.path.append("code/")
import unittest
from grid import Grid


class TestGridLoading(unittest.TestCase):
    """
    Test the loading of a grid from a file.
    """

    def test_grid0(self):
        """
        Grid 0 from file (colors + values) :
        """
        grid = Grid.grid_from_file("input/grid00.in", read_values=True)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(grid.value, [[5, 8, 4], [11, 1, 3]])

    def test_grid0_novalues(self):
        """
        Grid 0 from file (No values) :
        """
        grid = Grid.grid_from_file("input/grid00.in", read_values=False)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(
            grid.value, [[1, 1, 1], [1, 1, 1]]
        )  # Lorsque read_values=False, les cases ont toutes la valeur 1

    def test_grid1(self):
        """
        Grid 1 from file (colors + values) :
        """
        grid = Grid.grid_from_file("input/grid01.in", read_values=True)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 4, 3], [2, 1, 0]])
        self.assertEqual(grid.value, [[5, 8, 4], [11, 1, 3]])


if __name__ == "__main__":
    unittest.main()
