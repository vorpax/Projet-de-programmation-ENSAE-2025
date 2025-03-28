import sys
import os
import numpy as np
from grid import Grid


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# from color_grid_game import *


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

    def score(self) -> int:
        """
        Computes the total score of the current solution.

        The score is calculated as the sum of:
        - The values of remaining cells (cells that were not paired)
        - The costs of chosen pairs

        Returns:
        --------
        int
            The total score

        Time Complexity: O(n*m)
            Where n is the number of rows and m is the number of columns in the grid.
            The method iterates through all grid cells and pairs.
        """
        self.cells = [cell for pair in self.pairs for cell in pair]

        remaining_cells_cost = sum(
            self.grid.get_coordinate_value(cell[0], cell[1])
            for cell in self.all_cells
            if cell not in self.cells
            and self.grid.get_coordinate_color(cell[0], cell[1]) != "k"
        )
        chosen_pairs_cost = sum(self.grid.cost(pair) for pair in self.pairs)
        return remaining_cells_cost + chosen_pairs_cost


class SolverHungarianChibre(Solver):
    """
    An alternative implementation of the Hungarian algorithm solver.
    """

    def __init__(self, grid):
        super().__init__(grid)
        self.rules = "original rules"

    def run(self):
        """
        Builds a bipartite cost matrix using only cells present in valid pairs.
        Applies the Hungarian algorithm to find optimal pairs.

        Returns
        -------
        list of tuple
            A list of pairs of cells, each represented as a tuple of tuples.

        Raises
        ------
        ValueError
            If the cost matrix is empty or if pairs are invalid.
        """
        pairs = self.grid.all_pairs()
        all_cells = list(set(cell for pair in pairs for cell in pair))
        if self.rules == "original rules":
            # Split into even/odd based on coordinate parity
            even_cells = []
            odd_cells = []
            for cell in all_cells:
                if (cell[0] + cell[1]) % 2 == 0:
                    even_cells.append(cell)
                else:
                    odd_cells.append(cell)

            # Create mappings for matrix indices
            even_to_idx = {cell: i for i, cell in enumerate(even_cells)}
            odd_to_idx = {cell: j for j, cell in enumerate(odd_cells)}

            # Build cost matrix with valid pairs only and pad to square
            even_count = len(even_cells)
            odd_count = len(odd_cells)
            max_dim = max(even_count, odd_count)
            cost_matrix = np.zeros((max_dim, max_dim))
            for u, v in pairs:
                # Ensure u is even and v is odd
                if (u[0] + u[1]) % 2 != 0:
                    u, v = v, u
                if u in even_to_idx and v in odd_to_idx:
                    cost = self.grid.cost((u, v))
                    weight = (
                        cost - self.grid.value[u[0]][u[1]] - self.grid.value[v[0]][v[1]]
                    )
                    cost_matrix[even_to_idx[u], odd_to_idx[v]] = weight

            # Apply Hungarian algorithm on the padded square matrix
            row_ind, col_ind = self.linear_sum_assignment(cost_matrix)

            # Rebuild pairs from matrix indices, filtering valid entries
            self.pairs = []
            for i, j in zip(row_ind, col_ind):
                if i < even_count and j < odd_count and cost_matrix[i][j] != 0:
                    self.pairs.append((even_cells[i], odd_cells[j]))

        elif self.rules == "new rules":
            # Create a square cost matrix
            num_cells = len(all_cells)
            cost_matrix = np.zeros((num_cells, num_cells))

            # Create a mapping from cell to matrix index
            cell_to_idx = {cell: i for i, cell in enumerate(all_cells)}

            # Fill the cost matrix
            for u, v in pairs:
                if u in cell_to_idx and v in cell_to_idx:
                    cost = self.grid.cost((u, v))
                    weight = (
                        cost - self.grid.value[u[0]][u[1]] - self.grid.value[v[0]][v[1]]
                    )
                    cost_matrix[cell_to_idx[u], cell_to_idx[v]] = weight

            # Apply Hungarian algorithm on the square matrix
            row_ind, col_ind = self.linear_sum_assignment(cost_matrix)

            # Rebuild pairs from matrix indices, filtering valid entries
            self.pairs = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i][j] != 0:
                    self.pairs.append((all_cells[i], all_cells[j]))

        return self.pairs

    def linear_sum_assignment(self, cost, maximize=False):
        """
        Solve the linear sum assignment problem.
        Args:
            cost: The cost matrix of the bipartite graph.
            maximize: Calculates a maximum weight matching if true.
        Returns:
            An array of row indices and one of corresponding column indices giving the
            optimal assignment. The cost of the assignment can be computed as
            ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be sorted; in
            the case of a square cost matrix they will be equal to ``numpy.arange
            (cost_matrix.shape[0])``.
        """
        transpose = cost.shape[1] < cost.shape[0]

        if cost.shape[0] == 0 or cost.shape[1] == 0:
            return np.array([]), np.array([])

        if transpose:
            cost = cost.T

        cost = (-cost if maximize else cost).astype(float)

        u = np.full(cost.shape[0], 0.0, dtype=float)
        v = np.full(cost.shape[1], 0.0, dtype=float)
        path = np.full(cost.shape[1], -1, dtype=int)
        col4row = np.full(cost.shape[0], -1, dtype=int)
        row4col = np.full(cost.shape[1], -1, dtype=int)

        for current_row in range(cost.shape[0]):
            cost, u, v, path, row4col, col4row = self._lsa_body(
                cost, u, v, path, row4col, col4row, current_row
            )

        if transpose:
            v = col4row.argsort()
            return col4row[v], v
        else:
            return np.arange(cost.shape[0]), col4row

    def _find_short_augpath_while_body_inner_for(self, it, val):
        (
            remaining,
            min_value,
            cost,
            i,
            u,
            v,
            shortest_path_costs,
            path,
            lowest,
            row4col,
            index,
        ) = val

        j = remaining[it]
        r = min_value + cost[i, j] - u[i] - v[j]

        if r < shortest_path_costs[j]:
            path[j] = i
        shortest_path_costs[j] = min(shortest_path_costs[j], r)

        if (shortest_path_costs[j] < lowest) or (
            shortest_path_costs[j] == lowest and row4col[j] == -1
        ):
            index = it
        lowest = min(lowest, shortest_path_costs[j])

        return (
            remaining,
            min_value,
            cost,
            i,
            u,
            v,
            shortest_path_costs,
            path,
            lowest,
            row4col,
            index,
        )

    def _find_short_augpath_while_body_tail(self, val):
        remaining, index, row4col, sink, i, SC, num_remaining = val

        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        SC[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

        return remaining, index, row4col, sink, i, SC, num_remaining

    def _find_short_augpath_while_body(self, val):
        (
            cost,
            u,
            v,
            path,
            row4col,
            current_row,
            min_value,
            num_remaining,
            remaining,
            SR,
            SC,
            shortest_path_costs,
            sink,
        ) = val

        index = -1
        lowest = np.inf
        SR[current_row] = True

        for it in range(num_remaining):
            (
                remaining,
                min_value,
                cost,
                current_row,
                u,
                v,
                shortest_path_costs,
                path,
                lowest,
                row4col,
                index,
            ) = self._find_short_augpath_while_body_inner_for(
                it,
                (
                    remaining,
                    min_value,
                    cost,
                    current_row,
                    u,
                    v,
                    shortest_path_costs,
                    path,
                    lowest,
                    row4col,
                    index,
                ),
            )

        min_value = lowest
        if min_value == np.inf:
            sink = -1

        if sink == -1:
            remaining, index, row4col, sink, current_row, SC, num_remaining = (
                self._find_short_augpath_while_body_tail(
                    (remaining, index, row4col, sink, current_row, SC, num_remaining)
                )
            )

        return (
            cost,
            u,
            v,
            path,
            row4col,
            current_row,
            min_value,
            num_remaining,
            remaining,
            SR,
            SC,
            shortest_path_costs,
            sink,
        )

    def _find_short_augpath_while_cond(self, val):
        sink = val[-1]
        return sink == -1

    def _find_augmenting_path(self, cost, u, v, path, row4col, current_row):
        min_value = 0
        num_remaining = cost.shape[1]
        remaining = np.arange(cost.shape[1])[::-1]

        SR = np.full(cost.shape[0], False, dtype=bool)
        SC = np.full(cost.shape[1], False, dtype=bool)

        shortest_path_costs = np.full(cost.shape[1], np.inf)
        sink = -1

        while self._find_short_augpath_while_cond(
            (
                cost,
                u,
                v,
                path,
                row4col,
                current_row,
                min_value,
                num_remaining,
                remaining,
                SR,
                SC,
                shortest_path_costs,
                sink,
            )
        ):
            (
                cost,
                u,
                v,
                path,
                row4col,
                current_row,
                min_value,
                num_remaining,
                remaining,
                SR,
                SC,
                shortest_path_costs,
                sink,
            ) = self._find_short_augpath_while_body(
                (
                    cost,
                    u,
                    v,
                    path,
                    row4col,
                    current_row,
                    min_value,
                    num_remaining,
                    remaining,
                    SR,
                    SC,
                    shortest_path_costs,
                    sink,
                )
            )

        return sink, min_value, SR, SC, shortest_path_costs, path

    def _augment_previous_while_body(self, val):
        path, sink, row4col, col4row, current_row, _ = val

        i = path[sink]
        row4col[sink] = i

        col4row[i], sink = sink, col4row[i]
        breakvar = i == current_row

        return path, sink, row4col, col4row, current_row, breakvar

    def _augment_previous_while_cond(self, val):
        breakvar = val[-1]
        return not breakvar

    def _lsa_body(self, cost, u, v, path, row4col, col4row, current_row):
        sink, min_value, SR, SC, shortest_path_costs, path = self._find_augmenting_path(
            cost, u, v, path, row4col, current_row
        )

        u[current_row] += min_value
        mask = SR & (np.arange(cost.shape[0]) != current_row)
        u += mask * (min_value - shortest_path_costs[col4row])

        mask = SC
        v += mask * (shortest_path_costs - min_value)

        breakvar = False
        while self._augment_previous_while_cond(
            (path, sink, row4col, col4row, current_row, breakvar)
        ):
            path, sink, row4col, col4row, current_row, breakvar = (
                self._augment_previous_while_body(
                    (path, sink, row4col, col4row, current_row, breakvar)
                )
            )

        return cost, u, v, path, row4col, col4row
