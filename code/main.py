"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

from grid import Grid
from solver import SolverFulkerson


DATA_PATH = "./input/"

# Load a grid from file with values
FILE_NAME = DATA_PATH + "grid21.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)

# Display the grid
grid.plot()

# Initialize the grid cells
grid.cell_init()

print("Processing grid...")

# Create and run the Ford-Fulkerson solver
solver = SolverFulkerson(grid)
matching_pairs = solver.run()
print(f"Found {len(matching_pairs)} matching pairs")

print("Finished")
