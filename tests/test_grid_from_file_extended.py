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
        # On crée un tempfile
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")

        # On écrit dans le tempfile
        self.temp_file.write("2 3\n")  # (2,3)
        self.temp_file.write("0 1 2\n")  # Couleur 1 ligne
        self.temp_file.write("3 4 0\n")  # Couleur 2 ligne
        self.temp_file.write("10 20 30\n")  # Valeur 1 ligne
        self.temp_file.write("40 50 60\n")  # Valeur 2 ligne
        self.temp_file.close()

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

        # On check la dim
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)

        # On check la couleur
        self.assertEqual(grid.color, [[0, 1, 2], [3, 4, 0]])

        # On check les valeurs
        self.assertEqual(grid.value, [[10, 20, 30], [40, 50, 60]])

    def test_grid_from_file_without_values(self):
        """
        Test grid_from_file when not reading values.
        """
        grid = Grid.grid_from_file(self.file_path, read_values=False)

        # On check la dim
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)

        # On check la couleur
        self.assertEqual(grid.color, [[0, 1, 2], [3, 4, 0]])

        # On veut s'assurer que les valeurs sont bien à 1 (défaut)
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
        # On veut s'assurer que le code n'est pas d'accord (nrv)
        invalid_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        invalid_file.write("2 3\n")
        invalid_file.write("0 1\n")
        invalid_file.write("3 4 0\n")
        invalid_file.close()

        with self.assertRaises(ValueError):
            Grid.grid_from_file(invalid_file.name, read_values=False)

        os.unlink(invalid_file.name)

    def test_grid_from_file_invalid_color(self):
        """
        Test grid_from_file with invalid color value.
        """
        # On veut des mauvaises couleurs (et s'assurer que le code s'en plaigne)
        invalid_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
        invalid_file.write("2 3\n")
        invalid_file.write("0 1 7\n")
        invalid_file.write("3 4 0\n")
        invalid_file.close()

        with self.assertRaises(ValueError):
            Grid.grid_from_file(invalid_file.name, read_values=False)

        os.unlink(invalid_file.name)


if __name__ == "__main__":
    unittest.main()
