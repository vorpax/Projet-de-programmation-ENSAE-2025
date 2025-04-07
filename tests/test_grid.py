"""
Unit tests for the Grid class.
"""

import sys
import unittest
from io import StringIO
from unittest.mock import patch

sys.path.append("code/")
from grid import Grid


class TestGrid(unittest.TestCase):
    """
    Test the Grid class functionality.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """

        self.grid = Grid(2, 3)
        self.grid.color = [[0, 4, 3], [2, 1, 0]]
        self.grid.value = [[5, 8, 4], [11, 1, 3]]

    def test_init_default(self):
        """
        Test default initialization of grid.
        """
        grid = Grid(2, 3)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(grid.value, [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(grid.colors_list, ["w", "r", "b", "g", "k"])

    def test_init_with_values(self):
        """
        Test initialization with custom values.
        """
        colors = [[0, 4, 3], [2, 1, 0]]
        values = [[5, 8, 4], [11, 1, 3]]
        grid = Grid(2, 3, colors, values)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, colors)
        self.assertEqual(grid.value, values)

    def test_str_representation(self):
        """
        Test string representation of the grid.
        """

        # Erreur pas grave, j'ai juste changé le formating de la string représentant la grile.
        expected_output = "The grid is 2 x 3. It has the following colors:\n['w', 'k', 'g']\n['b', 'r', 'w']\nand the following values:\n[5, 8, 4]\n[11, 1, 3]\n"
        self.assertEqual(str(self.grid), expected_output)

    def test_repr_representation(self):
        """
        Test repr representation of the grid.
        """
        self.assertEqual(repr(self.grid), "<grid.Grid: n=2, m=3>")

    def test_is_forbidden(self):
        """
        Test is_forbidden method.
        """
        # Black cell
        self.grid.color[0][1] = 4  # Black cell at (0, 1)
        self.assertTrue(self.grid.is_forbidden(0, 1))
        # Non-black cell
        self.assertFalse(self.grid.is_forbidden(0, 0))
        # Out of bounds
        with self.assertRaises(IndexError):
            self.grid.is_forbidden(5, 5)

    def test_is_pair_forbidden(self):
        """
        Test is_pair_forbidden method.
        """
        # White-White pair (allowed)
        self.grid.color[0][0] = 0  # White
        self.grid.color[0][2] = 0  # White
        self.assertFalse(self.grid.is_pair_forbidden([(0, 0), (0, 2)]))

        # Black-White pair (forbidden)
        self.grid.color[0][0] = 4  # Black
        self.grid.color[0][2] = 0  # White
        self.assertTrue(self.grid.is_pair_forbidden([(0, 0), (0, 2)]))

        # Out of bounds
        with self.assertRaises(IndexError):
            self.grid.is_pair_forbidden([(5, 5), (0, 0)])

    def test_cost(self):
        """
        Test cost method.
        """
        # Value at (0, 0) is 5 and value at (1, 0) is 11
        self.assertEqual(self.grid.cost([(0, 0), (1, 0)]), 6)

        # If the pair is forbidden, the cost should be 0
        # Make the pair forbidden by making one cell black
        original_color = self.grid.color[0][0]
        self.grid.color[0][0] = 4  # Black
        self.assertEqual(self.grid.cost([(0, 0), (1, 0)]), 0)
        # Restore the original color
        self.grid.color[0][0] = original_color

    def test_all_pairs(self):
        """
        Test all_pairs method.
        """
        # Reset grid to have no forbidden pairs
        self.grid.color = [[0, 0, 0], [0, 0, 0]]
        # Should have 2*(2-1) + 3*(3-1) = 8 pairs (horizontal and vertical)
        pairs = self.grid.all_pairs()
        self.assertEqual(
            len(pairs), 7
        )  # 2 rows with 2 horizontal connections + 3 columns with 1 vertical connection

        # Make a cell black to create forbidden pairs
        self.grid.color[0][1] = 4  # Black
        pairs_with_black = self.grid.all_pairs()
        self.assertTrue(len(pairs_with_black) < len(pairs))

    def test_get_coordinate_color(self):
        """
        Test get_coordinate_color method.
        """
        self.assertEqual(self.grid.get_coordinate_color(0, 0), "w")  # White
        self.assertEqual(self.grid.get_coordinate_color(0, 1), "k")  # Black
        self.assertEqual(self.grid.get_coordinate_color(0, 2), "g")  # Green
        self.assertEqual(self.grid.get_coordinate_color(1, 0), "b")  # Blue
        self.assertEqual(self.grid.get_coordinate_color(1, 1), "r")  # Red

    def test_get_coordinate_value(self):
        """
        Test get_coordinate_value method.
        """
        self.assertEqual(self.grid.get_coordinate_value(0, 0), 5)
        self.assertEqual(self.grid.get_coordinate_value(0, 1), 8)
        self.assertEqual(self.grid.get_coordinate_value(0, 2), 4)
        self.assertEqual(self.grid.get_coordinate_value(1, 0), 11)
        self.assertEqual(self.grid.get_coordinate_value(1, 1), 1)

    def test_cell_init(self):
        """
        Test cell_init method.
        """
        self.grid.cell_init()
        self.assertEqual(len(self.grid.cells), 2)  # 2 rows
        self.assertEqual(len(self.grid.cells[0]), 3)  # 3 columns
        self.assertEqual(len(self.grid.cells_list), 6)  # Total 6 cells

        # Check one cell
        cell = self.grid.cells[0][0]
        self.assertEqual(cell.i, 0)
        self.assertEqual(cell.j, 0)
        self.assertEqual(cell.color, 0)
        self.assertEqual(cell.value, 5)


if __name__ == "__main__":
    unittest.main()
