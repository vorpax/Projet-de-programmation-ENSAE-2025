from grid import Grid
from solver import SolverGreedy

grid = Grid(2, 3)
print(grid)

data_path = "./input/"

file_name = data_path + "grid01.in"
grid = Grid.grid_from_file(file_name)
print(grid)

file_name = data_path + "grid01.in"
grid = Grid.grid_from_file(file_name, read_values=True)
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
