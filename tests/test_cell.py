"""
Unit tests for the Cell class.
"""

import sys
import unittest

sys.path.append("code/")
from grid import Cell


class TestCell(unittest.TestCase):
    """
    Test the Cell class functionality.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.cell = Cell(1, 2, 3, 4)  # i=1, j=2, color=3, value=4

    def test_init(self):
        """
        Test initialization of cell.
        """
        self.assertEqual(self.cell.i, 1)
        self.assertEqual(self.cell.j, 2)
        self.assertEqual(self.cell.color, 3)
        self.assertEqual(self.cell.value, 4)
        self.assertEqual(self.cell.ispair, 1)  # (1+2)%2 = 1

    def test_str_representation(self):
        """
        Test string representation of the cell.
        """
        expected_output = "Cell (1, 2) has color 3 and value 4"
        self.assertEqual(str(self.cell), expected_output)

    def test_repr_representation(self):
        """
        Test repr representation of the cell.
        """
        self.assertEqual(repr(self.cell), "<grid.Cell: i=1, j=2>")

    def test_parity_calculation(self):
        """
        Test that ispair attribute is correctly calculated.
        """
        # Even parity cells
        cell1 = Cell(0, 0, 0, 1)  # 0+0 = 0 (even)
        cell2 = Cell(1, 1, 0, 1)  # 1+1 = 2 (even)
        cell3 = Cell(2, 2, 0, 1)  # 2+2 = 4 (even)
        
        # Odd parity cells
        cell4 = Cell(0, 1, 0, 1)  # 0+1 = 1 (odd)
        cell5 = Cell(1, 0, 0, 1)  # 1+0 = 1 (odd)
        cell6 = Cell(2, 3, 0, 1)  # 2+3 = 5 (odd)
        
        self.assertEqual(cell1.ispair, 0)
        self.assertEqual(cell2.ispair, 0)
        self.assertEqual(cell3.ispair, 0)
        
        self.assertEqual(cell4.ispair, 1)
        self.assertEqual(cell5.ispair, 1)
        self.assertEqual(cell6.ispair, 1)


if __name__ == "__main__":
    unittest.main()