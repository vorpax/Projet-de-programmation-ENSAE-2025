"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

import sys
from gridArthur import Grid as GridArthur
from grid import Grid
from solver import SolverHungarian
from grid import Grid

# from solverArthur import SolverGeneral

# from max_weight_matching import max_weight_matching


def main():
    """
    Main function that loads a grid from a file and runs the Hungarian solver.

    Usage:
        python main.py [grid_file_path]

    If no grid file path is provided, it defaults to grid01.in
    """
    # Handle command line arguments for grid file
    data_path = "./input/"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = data_path + "grid00.in"

    # Load grid from file
    grid = Grid.grid_from_file(file_name, read_values=True)
    grid.cell_init()

    # Display grid information
    print_grid_cells(grid)

    # Initialize and run the Hungarian solver
    solver = SolverHungarian(grid)

    # Display valid pair information
    print_valid_pairs(grid)

    # Run the algorithm
    matching_pairs = solver.run()
    score = solver.score()

    # Display results
    print_results(grid, solver, matching_pairs, score)


def print_grid_cells(grid):
    """Print all cells in the grid with their colors and values."""
    for i in range(grid.n):
        for j in range(grid.m):
            color = grid.get_coordinate_color(i, j)
            value = grid.get_coordinate_value(i, j)
            print(f"Cell ({i},{j}): color={color}, value={value}")


def print_valid_pairs(grid):
    """Print information about all possible valid pairs in the grid."""
    print("All possible valid pairs for this grid:")
    all_valid_pairs = []

    print("Grid colors and values:")
    for i in range(grid.n):
        color_row = [grid.color[i][j] for j in range(grid.m)]
        value_row = [grid.value[i][j] for j in range(grid.m)]
        print(f"Row {i}: Colors={color_row}, Values={value_row}")

    print("\nPossible pairs:")
    for i in range(grid.n):
        for j in range(grid.m):
            # Check adjacent cells (up, down, left, right)
            adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for adj_i, adj_j in adjacents:
                if 0 <= adj_i < grid.n and 0 <= adj_j < grid.m:
                    pair = ((i, j), (adj_i, adj_j))
                    if not grid.is_pair_forbidden(pair):
                        cost = grid.cost(pair)
                        cell1_value = grid.get_coordinate_value(i, j)
                        cell2_value = grid.get_coordinate_value(adj_i, adj_j)
                        expected_cost = abs(cell1_value - cell2_value)
                        all_valid_pairs.append((pair, cost))
                        print(
                            f"Pair {pair}: Cost = {cost}, Expected = {expected_cost}, "
                            f"Cell1 value = {cell1_value}, Cell2 value = {cell2_value}"
                        )

    print(f"\nTotal possible valid pairs: {len(all_valid_pairs)}\n")
    print("=" * 50)


def print_results(grid, solver, matching_pairs, score):
    """Print the results of the solver run."""
    print(f"Final score: {score}")
    print(f"Number of pairs: {len(matching_pairs)}")

    # Get costs for each pair
    pair_costs = []
    for pair in matching_pairs:
        cost = grid.cost(pair)
        pair_costs.append((pair, cost))

    print("\nAll paired cells and their costs:")
    for i, (pair, cost) in enumerate(pair_costs):
        print(f"Pair {i+1}: {pair} - Cost: {cost}")

    # Calculate score components
    remaining_cells = [cell for cell in solver.all_cells if cell not in solver.cells]
    remaining_cells_cost = sum(
        grid.get_coordinate_value(cell[0], cell[1])
        for cell in remaining_cells
        if grid.get_coordinate_color(cell[0], cell[1]) != "k"
    )
    chosen_pairs_cost = sum(grid.cost(pair) for pair in matching_pairs)

    print(f"\nScore breakdown:")
    print(f"- Remaining cells cost: {remaining_cells_cost}")
    print(f"- Chosen pairs cost: {chosen_pairs_cost}")
    print(f"- Total score: {remaining_cells_cost + chosen_pairs_cost}")

    print(f"\nNumber of remaining cells: {len(remaining_cells)}")
    if remaining_cells:
        print("All remaining cells:")
        for cell in remaining_cells:
            value = grid.get_coordinate_value(cell[0], cell[1])
            color = grid.get_coordinate_color(cell[0], cell[1])
            print(f"Cell {cell}: Value = {value}, Color = {color}")


if __name__ == "__main__":
    # main()
    data_path = "./input/"
    file_name = data_path + "grid00.in"

    # Load grid from file
    grid = Grid.grid_from_file(file_name, read_values=True)
    # grid.cell_init()

    solver = SolverHungarian(grid)
    solver.run()
    print(solver.score())
    print("brk")

# Hungarian Algorithm
# Edmond's algorithm
# edmond's algorithm
# https://cp-algorithms.com/graph/Assignment-problem-min-flow.html
# https://github.com/keon/algorithms
# Va falloir mettre de bons poids pour gérer le truc.
# Mettre un max valeur à la bien.
