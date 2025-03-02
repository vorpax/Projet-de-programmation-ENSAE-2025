"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

from grid import Grid
from solver import SolverFulkerson

DATA_PATH = "./input/"

FILE_NAME = DATA_PATH + "grid21.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)

grid.plot()

grid.cell_init()

solver = SolverFulkerson(grid)
matching_pairs = solver.run()
print(f"Found {len(matching_pairs)} matching pairs")

print("RHA OUAI")
