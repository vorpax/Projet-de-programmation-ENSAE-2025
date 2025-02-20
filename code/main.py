"""
a simple main file to test the grid and the solver
"""

from grid import Grid, Cell


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


# Adjacency dictionnary which is the ground for modeling the graph
dict_adjacency = {}
dict_adjacency["source"] = [
    Cell for Cell in grid.cells_list if not (Cell.i + Cell.j) % 2
]
dict_adjacency["puit"] = [cell for cell in grid.cells_list if (cell.i + cell.j) % 2]


def add_children(cell: Cell):
    """
    add the children of a cell to the adjacency dictionnary
    """
    if dict_adjacency.get(f"cell_{cell.i}_{cell.j}") is None:
        dict_adjacency[f"cell_{cell.i}_{cell.j}"] = []

    adjecente = [[cell.i + 1, cell.j], [cell.i - 1, cell.j], [cell.i, cell.j + 1]]

    for cell_adjecente in adjecente:
        if cell_adjecente[0] in range(grid.n):
            if cell_adjecente[1] in range(grid.m):
                if not grid.is_pair_forbidden([[cell.i, cell.j], cell_adjecente]):
                    dict_adjacency[f"cell_{cell.i}_{cell.j}"].append(
                        grid.cells[cell_adjecente[0]][cell_adjecente[1]]
                    )
                if (
                    dict_adjacency.get(f"cell_{cell_adjecente[0]}_{cell_adjecente[1]}")
                    is None
                ):
                    add_children(grid.cells[cell_adjecente[0]][cell_adjecente[1]])

    # if cell_adjecente[0] in range(grid.n):
    #         if cell_adjecente[1] in range(grid.m):


for child_cell in dict_adjacency["source"]:
    add_children(child_cell)

print("break")
