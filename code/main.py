"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

from grid import Grid
from solver import SolverHungarian

DATA_PATH = "./input/"

FILE_NAME = DATA_PATH + "grid05.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)

# Check the raw grid values
print("Raw grid values:")
print(f"Colors: {grid.color}")
print(f"Values: {grid.value}")

grid.cell_init()

# Check that cell values were initialized correctly
print("Cell values after initialization:")
for i in range(grid.n):
    for j in range(grid.m):
        color = grid.get_coordinate_color(i, j)
        value = grid.get_coordinate_value(i, j)
        print(f"Cell ({i},{j}): color={color}, value={value}")


# solver = SolverFulkerson(grid)
# matching_pairs = solver.run()
# print(f"Found {len(matching_pairs)} matching pairs")

# print("RHA OUAI")

# saucisse = {"lezgongue": "1", "b": "2", "c": "3"}
# print(list(saucisse))
# saucisse["d"] = "4"
# saucisse["a"] = "lezgongue"
# print(list(saucisse))

# dict_list = [(clee, saucisse[clee]) for clee in list(saucisse)]

# print(dict_list)

# Plutôt des algorithmes de weighted matching (fallait pas voir car très compliqué mais, c'est mieux que la Hongrie)


GrandPereExplorer = SolverHungarian(grid)
# GrandPereExplorer.adjacency_dict_init()

# First, check all possible valid pairs and their costs
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
                        f"Pair {pair}: Cost = {cost}, Expected = {expected_cost}, Cell1 value = {cell1_value}, Cell2 value = {cell2_value}"
                    )

print(f"\nTotal possible valid pairs: {len(all_valid_pairs)}\n")
print("=" * 50)

# Run the algorithm
matching_pairs = GrandPereExplorer.run()
score = GrandPereExplorer.score()
print(f"Final score: {score}")
print(f"Number of pairs: {len(matching_pairs)}")

# Debug - check individual pair costs
pair_costs = []
for pair in matching_pairs:
    cost = grid.cost(pair)
    pair_costs.append((pair, cost))

print("\nAll paired cells and their costs:")
for i, (pair, cost) in enumerate(pair_costs):
    print(f"Pair {i+1}: {pair} - Cost: {cost}")

# Calculate the components of the score
remaining_cells = [
    cell for cell in GrandPereExplorer.all_cells if cell not in GrandPereExplorer.cells
]
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

print("j'habite une maison citrouille")
# afficher_matrice()

# la_hongrie()

print("j'habite une maison citrouille")
