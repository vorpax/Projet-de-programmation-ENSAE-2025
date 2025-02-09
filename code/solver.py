"""
A module for the solver class and implementations of it.
"""

from grid import Grid


class Solver:
    """
    A solver class.

    Attributes:
    -----------
    grid: Grid
        The grid
    pairs: list[tuple[tuple[int]]]
        A list of pairs, each being a tuple ((i1, j1), (i2, j2))
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

    def score(self):
        """
        Computes the of the list of pairs in self.pairs
        """
        return sum(self.grid.cost(pair) for pair in self.pairs)


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

        return chosen_pairs
