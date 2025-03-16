"""
Main module for testing the grid and solvers.

This module loads a grid from an input file, initializes cells, and runs
a solver to find a matching solution. It demonstrates the basic workflow
of loading a grid and applying a solver algorithm.
"""

from grid import Grid
from solver import SolverHungarian

DATA_PATH = "./input/"

FILE_NAME = DATA_PATH + "grid01.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)

# grid.plot()

grid.cell_init()


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
GrandPereExplorer.adjacency_dict_init()
print("j'habite une maison citrouille")


def la_hongrie():
    """
    Runs the Hungarian algorithm to find the optimal matching.
    """

    cost_matrix = GrandPereExplorer.hungarian_algorithm()
    print(cost_matrix)
    print("jroule que en CR jpeux pas trahier Honda")


def afficher_matrice():
    """
    Displays the adjacency matrix of the grid, weighted by the cost.
    """

    listeazeubi = GrandPereExplorer.dict_adjacency.values()

    for i in listeazeubi:
        print(list(i.values()))


# afficher_matrice()

la_hongrie()

print("j'habite une maison citrouille")
