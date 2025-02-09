"""
a simple main file to test the grid and the solver
"""

from .grid import Grid
from .solver import SolverGreedy

grid = Grid(2, 3)
print(grid)

DATA_PATH = "./input/"

FILE_NAME = DATA_PATH + "grid01.in"
grid = Grid.grid_from_file(FILE_NAME)
print(grid)

FILE_NAME = DATA_PATH + "grid01.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)
print(grid)


AllPairs = grid.all_pairs()
AllValues = [grid.cost(pairs) for pairs in AllPairs]
print(AllValues)

# grid.plot()


solver = SolverGreedy(grid)

GreedyPairs = solver.run()
print(GreedyPairs)
GreedyPairsCost = [grid.cost(pairs) for pairs in GreedyPairs]


print(GreedyPairsCost)
print("e")
print("The final score of SolverEmpty is:", solver.score())
