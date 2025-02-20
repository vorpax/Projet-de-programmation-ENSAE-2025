"""
A module for the solver class and implementations of it.
"""

from code.grid import Grid, Cell


class Solver:
    """
    A solver class.

    Attributes:
    -----------
    grid: Grid
        The grid
    pairs: list[tuple[tuple[int]]]
        A list of chosen pairs, each being a tuple ((i1, j1), (i2, j2)). Those are chosen after running the solver.
    cells: list[tuple[int]] :
        A list of chosen cells, each being a tuple (i, j). Those are added after running the solver.
    all_cells: list[tuple[int]]
        A list of every cells of the grid, each being a tuple (i, j).
    """

    def __init__(self, grid: Grid):
        """
        Initializes the solver.

        Parameters:
        -----------
        grid: Grid
            The grid
        """
        self.grid = grid
        self.pairs = []
        self.cells = []
        self.all_cells = [
            (i, j) for i in range(self.grid.n) for j in range(self.grid.m)
        ]

    def score(self):
        """
        Computes the of the list of pairs in self.pairs
        """

        remaining_cells_cost = sum(
            self.grid.value(cell[0], cell[1])
            for cell in self.all_cells
            if cell not in self.cells
        )
        chosen_pairs_cost = sum(self.grid.cost(pair) for pair in self.pairs)
        return remaining_cells_cost + chosen_pairs_cost


class SolverEmpty(Solver):
    """
    An empty solver for testing purposes
    """

    def run(self):
        """
        Returns an empty list of pairs (Empty solver).
        """


class SolverGreedy(Solver):
    """
    A greedy solver that chooses the cheapest pair at each step
    """

    def run(self):
        """
        Runs the greedy solver. Returns the list of each pair chosen.
        """

        chosen_pairs = self.pairs
        chosen_cells = self.cells
        all_pairs_sorted = self.grid.all_pairs().copy()
        all_pairs_sorted.sort(key=self.grid.cost)
        i = 0

        while len(all_pairs_sorted) > 0:
            i += 1
            filtered_list = []
            for pair in all_pairs_sorted:
                if pair[0] not in chosen_cells and pair[1] not in chosen_cells:
                    filtered_list.append(pair)

            if len(filtered_list) != 0:
                cheapest_pair = filtered_list.pop(0)
                chosen_pairs.append(cheapest_pair)
                chosen_cells.append(cheapest_pair[0])
                chosen_cells.append(cheapest_pair[1])
                cost_cheapest_pair = self.grid.cost(cheapest_pair)
                print(
                    f"The cost of the chosen pair (which is the cheapest) is : {cost_cheapest_pair}"
                )

            all_pairs_sorted = filtered_list

        self.pairs = chosen_pairs  # !!!! Va falloir modifier d'autres trucs puisque la méthode solver.run() n'est plus stateless

        for pair in self.pairs:
            self.cells.append(pair[0])
            self.cells.append(pair[1])

        return chosen_pairs


class SolverBasicMatching(Solver):
    """
    A solver thats based on graph theory and Ford-Fulkerson algorithm
    """

    def __init__(self, grid):
        """
        Initializes the solver
        """
        super().__init__(grid)
        self.dict_adjacency = {}
        self.adjacency_graph_init()

    def adjacency_graph_init(self):
        """
        Initializes the adjacency graph of the grid
        """

        self.dict_adjacency["puit"] = [
            cell for cell in self.grid.cells_list if (cell.i + cell.j) % 2
        ]

        self.dict_adjacency["source"] = [
            Cell for Cell in self.grid.cells_list if not (Cell.i + Cell.j) % 2
        ]

        def add_children(cell: Cell):
            """
            add the children of a cell to the adjacency dictionnary
            """
            if self.dict_adjacency.get(f"cell_{cell.i}_{cell.j}") is None:
                self.dict_adjacency[f"cell_{cell.i}_{cell.j}"] = []

            adjecente = [
                [cell.i + 1, cell.j],
                [cell.i - 1, cell.j],
                [cell.i, cell.j + 1],
            ]

            for cell_adjecente in adjecente:
                if cell_adjecente[0] in range(self.grid.n):
                    if cell_adjecente[1] in range(self.grid.m):
                        if not self.grid.is_pair_forbidden(
                            [[cell.i, cell.j], cell_adjecente]
                        ):
                            self.dict_adjacency[f"cell_{cell.i}_{cell.j}"].append(
                                self.grid.cells[cell_adjecente[0]][cell_adjecente[1]]
                            )
                        if (
                            self.dict_adjacency.get(
                                f"cell_{cell_adjecente[0]}_{cell_adjecente[1]}"
                            )
                            is None
                        ):
                            add_children(
                                self.grid.cells[cell_adjecente[0]][cell_adjecente[1]]
                            )

            # if cell_adjecente[0] in range(self.grid.n):
            #         if cell_adjecente[1] in range(self.grid.m):

        for child_cell in self.dict_adjacency["source"]:
            add_children(child_cell)
