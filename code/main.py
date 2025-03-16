"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

from grid import Grid
from solver import SolverHungarian, SolverFulkerson

DATA_PATH = "./input/"


def run_hungarian_solver(grid_file):
    """
    Run the Hungarian algorithm solver on a grid.

    Parameters:
    -----------
    grid_file : str
        The path to the grid file
    """
    print(f"Running Hungarian solver on {grid_file}")
    grid = Grid.grid_from_file(grid_file, read_values=True)
    print(grid)

    # Initialize the Hungarian solver
    solver = SolverHungarian(grid)

    # Run the solver
    matching_pairs = solver.run()

    # Print results
    print(f"Hungarian algorithm found {len(matching_pairs)} pairs")
    print(f"Total score: {solver.score()}")

    # Optionally, uncomment to display the grid
    # grid.plot()

    return matching_pairs, solver.score()


def run_fulkerson_solver(grid_file):
    """
    Run the Ford-Fulkerson algorithm solver on a grid for comparison.

    Parameters:
    -----------
    grid_file : str
        The path to the grid file
    """
    print(f"Running Ford-Fulkerson solver on {grid_file}")
    grid = Grid.grid_from_file(grid_file, read_values=True)
    print(grid)

    # Initialize the Ford-Fulkerson solver
    solver = SolverFulkerson(grid)

    # Run the solver
    matching_pairs = solver.run()

    # Print results
    print(f"Ford-Fulkerson algorithm found {len(matching_pairs)} pairs")
    print(f"Total score: {solver.score()}")

    return matching_pairs, solver.score()


def compare_solvers(grid_file):
    """
    Compare the Hungarian and Ford-Fulkerson solvers on the same grid.

    Parameters:
    -----------
    grid_file : str
        The path to the grid file
    """
    print(f"Comparing solvers on {grid_file}")
    print("-" * 50)

    # Run both solvers
    hungarian_pairs, hungarian_score = run_hungarian_solver(grid_file)
    print("-" * 50)
    fulkerson_pairs, fulkerson_score = run_fulkerson_solver(grid_file)

    # Compare results
    print("\nComparison:")
    print(f"Hungarian pairs: {len(hungarian_pairs)}, Score: {hungarian_score}")
    print(f"Fulkerson pairs: {len(fulkerson_pairs)}, Score: {fulkerson_score}")

    if hungarian_score < fulkerson_score:
        print("Hungarian algorithm achieved a better (lower) score!")
    elif hungarian_score > fulkerson_score:
        print("Ford-Fulkerson algorithm achieved a better (lower) score!")
    else:
        print("Both algorithms achieved the same score.")


if __name__ == "__main__":
    # Test with a specific grid file
    grid_file = DATA_PATH + "grid01.in"

    # Uncomment to run just the Hungarian solver
    run_hungarian_solver(grid_file)

    # Uncomment to compare both solvers
    #
