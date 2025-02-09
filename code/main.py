"""
a simple main file to test the grid and the solver
"""

from grid import Grid
import matplotlib.pyplot as plt


grid = Grid(2, 3)
print(grid)

DATA_PATH = "./input/"

FILE_NAME = DATA_PATH + "grid01.in"
grid = Grid.grid_from_file(FILE_NAME)

FILE_NAME = DATA_PATH + "grid01.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)


grid.plot()
