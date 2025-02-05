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
        self.pairs = list()

    def score(self):
        """
        Computes the of the list of pairs in self.pairs
        """
        return "Method not implemented yet"


class SolverEmpty(Solver):
    """
    An empty solver for testing purposes
    """

    def run(self):
        """ """
        pass


class SolverGreedy(Solver):
    def run(self):
        pairs = []
        lowest_score_pair = self.grid.all_pairs().copy().sort(key=self.grid.cost)

        while len(lowest_score_pair) > 0:
            filtered_list = [
                pair
                for pair in lowest_score_pair
                if pair[0] not in lowest_score_pair and pair[1] not in lowest_score_pair
            ]
            lowest_score_pair = filtered_list
            pairs.append(filtered_list.pop())
        return pairs
