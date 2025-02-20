"""
a simple main file to test the grid and the solver
"""

from grid import Grid


grid = Grid(2, 3)
print(grid)

DATA_PATH = "./input/"

# FILE_NAME = DATA_PATH + "grid01.in"
# grid = Grid.grid_from_file(FILE_NAME)

FILE_NAME = DATA_PATH + "grid02.in"
grid = Grid.grid_from_file(FILE_NAME, read_values=True)
# grid.plot()
grid.cell_init()

print("poulet")


first_cell = grid.cells_list[0]

dico_machin = {}
dico_machin["source"] = []
dico_machin["puit"] = []


for i in range(grid.n):
    if not i % 2:
        dico_machin["source"].append(grid.cells[i][0])

for cell in dico_machin["source"]:
    dico_machin[f"cell_{cell.i}_{cell.j}"] = []

    adjecente = [[cell.i + 1, cell.j], [cell.i - 1, cell.j], [cell.i, cell.j + 1]]
    for cell_adjecente in adjecente:

        if cell_adjecente[0] in range(grid.n):
            if cell_adjecente[1] in range(grid.m):

                dico_machin[f"cell_{cell.i}_{cell.j}"].append(
                    grid.cells[cell_adjecente[0]][cell_adjecente[1]]
                )

print(dico_machin)

print("break")
