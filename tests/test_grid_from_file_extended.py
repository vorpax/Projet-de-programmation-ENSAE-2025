"""
Extended unit tests for the grid_from_file method in the Grid class.
"""

import sys
import unittest
import os
import tempfile

sys.path.append("code/")
from grid import Grid


class TestGridFromFileExtended(unittest.TestCase):
    """
    Additional test cases for the grid_from_file method.
    """

    def setUp(self):
        """
        Create a temporary test grid file.
        """
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        
        # Write a valid grid to the file
        self.temp_file.write("2 3\n")  # 2 rows, 3 columns
        self.temp_file.write("0 1 2\n")  # First row colors
        self.temp_file.write("3 4 0\n")  # Second row colors
        self.temp_file.write("10 20 30\n")  # First row values
        self.temp_file.write("40 50 60\n")  # Second row values
        self.temp_file.close()
        
        # Path to the file
        self.file_path = self.temp_file.name

    def tearDown(self):
        """
        Clean up temporary files.
        """
        if os.path.exists(self.file_path):
            os.unlink(self.file_path)

    def test_grid_from_file_with_values(self):
        """
        Test grid_from_file when reading values.
        """
        grid = Grid.grid_from_file(self.file_path, read_values=True)
        
        # Check grid dimensions
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        
        # Check colors
        self.assertEqual(grid.color, [[0, 1, 2], [3, 4, 0]])
        
        # Check values
        self.assertEqual(grid.value, [[10, 20, 30], [40, 50, 60]])

    def test_grid_from_file_without_values(self):
        """
        Test grid_from_file when not reading values.
        """
        grid = Grid.grid_from_file(self.file_path, read_values=False)
        
        # Check grid dimensions
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        
        # Check colors
        self.assertEqual(grid.color, [[0, 1, 2], [3, 4, 0]])
        
        # Check that values are all 1 when not reading them
        self.assertEqual(grid.value, [[1, 1, 1], [1, 1, 1]])

    def test_grid_from_file_nonexistent(self):
        """
        Test grid_from_file with a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            Grid.grid_from_file("non_existent_file.txt")

    def test_grid_from_file_invalid_format(self):
        """
        Test grid_from_file with invalid format.
        """
        # Create a temporary file with invalid format
        invalid_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        invalid_file.write("2 3\n")  # 2 rows, 3 columns
        invalid_file.write("0 1\n")  # Missing a column in the first row
        invalid_file.write("3 4 0\n")  # Second row colors
        invalid_file.close()
        
        # Test with the invalid file
        with self.assertRaises(ValueError):
            Grid.grid_from_file(invalid_file.name, read_values=False)
        
        # Clean up
        os.unlink(invalid_file.name)

    def test_grid_from_file_invalid_color(self):
        """
        Test grid_from_file with invalid color value.
        """
        # Create a temporary file with invalid color
        invalid_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        invalid_file.write("2 3\n")  # 2 rows, 3 columns
        invalid_file.write("0 1 7\n")  # 7 is an invalid color (valid is 0-4)
        invalid_file.write("3 4 0\n")  # Second row colors
        invalid_file.close()
        
        # Test with the invalid file
        with self.assertRaises(ValueError):
            Grid.grid_from_file(invalid_file.name, read_values=False)
        
        # Clean up
        os.unlink(invalid_file.name)


if __name__ == "__main__":
    unittest.main()