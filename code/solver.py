"""
A module for the solver class and its implementations.

This module contains the base Solver class and various solver implementations
including SolverEmpty, SolverGreedy, SolverFulkerson (which uses the
Ford-Fulkerson algorithm for maximum bipartite matching), and SolverHungarian
(which uses the Hungarian algorithm for minimum-cost bipartite matching).
"""

from math import inf
import numpy as np
from grid import Grid


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
        remaining_cells_cost = sum(
            self.grid.get_coordinate_value(cell[0], cell[1])
            for cell in self.all_cells
            if cell not in self.cells
            and self.grid.get_coordinate_color(cell[0], cell[1]) != "k"
        )
        chosen_pairs_cost = sum(self.grid.cost(pair) for pair in self.pairs)
        return remaining_cells_cost + chosen_pairs_cost


class SolverEmpty(Solver):
    """
    An empty solver for testing purposes.

    This solver doesn't select any pairs and returns an empty list.
    """

    def run(self) -> list:
        """
        Runs the empty solver.

        Returns:
        --------
        list
            An empty list of pairs
        """
        return []


class SolverGreedy(Solver):
    """
    A greedy solver that chooses the cheapest pair at each step.

    This implementation sorts all possible pairs by cost and iteratively
    selects the cheapest available pair, making sure not to reuse any cells.
    """

    def run(self) -> list[list[tuple[int, int]]]:
        """
        Runs the greedy solver.

        The algorithm:
        1. Sort all valid pairs by cost
        2. Iteratively select the cheapest pair that doesn't reuse cells
        3. Update the list of chosen pairs and cells

        Returns:
        --------
        list[list[tuple[int, int]]]
            The list of chosen pairs in the format [[(i1, j1), (i2, j2)], ...]

        Time Complexity: O(p^2)
            Where p is the number of valid pairs in the grid (which is O(n*m) in the worst case).
            The algorithm requires sorting all pairs O(p log p) and then for each pair,
            it may need to check against all previous pairs O(p).
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

        self.pairs = chosen_pairs

        self.cells = []
        for pair in self.pairs:
            self.cells.append(pair[0])
            self.cells.append(pair[1])

        return chosen_pairs




class SolverFulkerson(Solver):
    """
    A solver implementing the Ford-Fulkerson algorithm for maximum bipartite matching.

    This solver uses a network flow approach to find the maximum matching between cells
    of the grid. It models the problem as a bipartite graph where:
    - Even cells (i+j is even) are connected to a source node
    - Odd cells (i+j is odd) are connected to a sink node
    - Even cells are connected to adjacent odd cells if the pair is valid

    Attributes:
    -----------
    grid: Grid
        The grid to solve
    residual_graph: dict
        Dictionary representing the residual graph for the Ford-Fulkerson algorithm
    pairs: list
        The chosen pairs after running the algorithm
    cells: list
        The chosen cells after running the algorithm
    """

    def __init__(self, grid: Grid) -> None:
        """
        Initializes the solver with a grid and sets up the residual graph.

        Parameters:
        -----------
        grid: Grid
            The grid to solve
        """
        super().__init__(grid)
        self.residual_graph: dict = {}  # Dict 4 residual graph
        self.adjacency_graph_init()

    def adjacency_graph_init(self) -> None:
        """
        Initializes the residual graph for the Ford-Fulkerson algorithm.

        This method creates a graph representation where:
        1. A source node is connected to all even-parity cells
        2. All odd-parity cells are connected to a sink node
        3. Even cells are connected to adjacent odd cells if the pair is not forbidden

        All edges in the initial graph have capacity 1, representing the possibility
        of including that connection in the matching.

        Returns:
        --------
        None

        Time Complexity: O(n*m)
            Where n is the number of rows and m is the number of columns in the grid.
            The method processes every cell and its valid neighbors.
        """
        # On initialise source et sink (cellules "fictives", font pas partie de la grille)
        self.residual_graph["source"] = {}
        self.residual_graph["sink"] = {}

        # Identify even and odd cells
        even_cells = []
        odd_cells = []
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                cell_id = f"cell_{i}_{j}"
                # Faut bien commencer par initialiser le graphe résiduel pr sink source
                self.residual_graph[cell_id] = {}

                # Sort cells into even and odd based on i+j parity
                if (i + j) % 2 == 0:
                    even_cells.append((i, j))
                    # Capa de 1 car : une paire ne peut matcher qu'avec 1 seule autre paire
                    self.residual_graph["source"][cell_id] = 1
                else:
                    odd_cells.append((i, j))
                    # Cf ci-dessus
                    self.residual_graph[cell_id]["sink"] = 1

        # initialise arêtes cellules paires -> impaires selon contraintes
        for i, j in even_cells:
            cell_id = f"cell_{i}_{j}"
            # Haut bas droite gauche
            adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

            for adj_i, adj_j in adjacents:  # On unpack
                if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                    adj_cell_id = f"cell_{adj_i}_{adj_j}"

                    pair = [(i, j), (adj_i, adj_j)]
                    if not self.grid.is_pair_forbidden(pair):
                        self.residual_graph[cell_id][adj_cell_id] = 1

    def find_augmenting_path(self) -> list[str] | None:
        """
        Finds an augmenting path from source to sink in the residual graph using BFS.

        This is a key component of the Ford-Fulkerson algorithm. An augmenting path
        is a path from source to sink through the residual graph with available capacity
        on all edges.

        Returns:
        --------
        list[str] | None
            The path as a list of node IDs (strings), or None if no path exists

        Note:
        -----
        The implementation uses a simple list as a queue. For better performance,
        consider using collections.deque instead.

        Time Complexity: O(V + E)
            Where V is the number of vertices (cells + source + sink) and E is the number of edges
            in the residual graph. This is the standard complexity of BFS.
        """
        # Une queue
        queue = ["source"]
        # On veut garder la trace des noeuds visités, faudrait pas se perdre
        visited = {"source": None}

        while queue:
            current = queue.pop(0)

            # Si on a atteint le sink, on est bon
            if current == "sink":
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                path.reverse()  # Là, on passe du LIFO au FIFO
                return path

            if current in self.residual_graph:
                for neighbor, capacity in self.residual_graph[current].items():
                    if neighbor not in visited and capacity > 0:
                        visited[neighbor] = current
                        queue.append(neighbor)

        # On a pas trouvé de chemin :(
        return None

    def ford_fulkerson(self) -> int:
        """
        Implements the Ford-Fulkerson algorithm to find maximum flow/matching.

        This algorithm:
        1. Repeatedly finds augmenting paths from source to sink
        2. Updates the residual graph by decreasing capacity in the forward direction
           and increasing capacity in the backward direction
        3. Continues until no more augmenting paths can be found

        Returns:
        --------
        int
            The maximum flow value, which equals the size of the maximum matching

        Time Complexity: O(V * E^2)
            Where V is the number of vertices (cells + source + sink) and E is the number of edges
            in the residual graph. In the worst case, each augmenting path search is O(E),
            and we may need to find up to O(V*E) augmenting paths.
        """

        max_flow = 0

        # Trouver flot augmentant et maj graphe résiduel
        path = self.find_augmenting_path()
        while path:
            # On veut identifier le bottleneck
            min_capacity = float("inf")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                min_capacity = min(min_capacity, self.residual_graph[u][v])

            # maj capa résiduelle
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                self.residual_graph[u][v] -= min_capacity

                # Prise en compte de la "capacité inverse" (prévenir de potentielles erreurs, prendre en compte annulations)
                if v not in self.residual_graph:
                    self.residual_graph[v] = {}
                if u not in self.residual_graph[v]:
                    self.residual_graph[v][u] = 0
                self.residual_graph[v][u] += min_capacity

            max_flow += min_capacity

            path = self.find_augmenting_path()  # Continue finding augmenting paths

        return max_flow

    def run(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Runs the solver and returns the matching pairs.

        This method:
        1. Executes the Ford-Fulkerson algorithm to find the maximum matching
        2. Extracts the matching by checking for backward edges with positive capacity
           (which indicate that flow passed through that edge)
        3. Converts the matching from the graph representation back to grid coordinates

        Returns:
        --------
        list[tuple[tuple[int, int], tuple[int, int]]]
            A list of matched pairs, where each pair is represented as ((i1, j1), (i2, j2))

        Time Complexity: O(V * E^2 + n*m)
            Where V is the number of vertices, E is the number of edges in the residual graph,
            n is the number of rows, and m is the number of columns in the grid.
            The dominant term is the Ford-Fulkerson algorithm O(V * E^2), plus the cost of
            extracting the matching from the graph O(n*m).
        """

        max_flow = self.ford_fulkerson()
        print(f"Maximum flow: {max_flow}")

        # On extrait matching à partir du graphe
        matching_pairs = []

        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i + j) % 2 == 0:  # Only check even cells
                    cell_id = f"cell_{i}_{j}"

                    # Haut bas droite gauche
                    adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

                    for adj_i, adj_j in adjacents:
                        if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                            adj_cell_id = f"cell_{adj_i}_{adj_j}"

                            if (
                                adj_cell_id in self.residual_graph
                                and cell_id in self.residual_graph[adj_cell_id]
                                and self.residual_graph[adj_cell_id][cell_id] > 0
                            ):
                                matching_pairs.append(((i, j), (adj_i, adj_j)))

                                self.cells.append((i, j))
                                self.cells.append((adj_i, adj_j))

        self.pairs = matching_pairs
        return matching_pairs


class SolverHungarian(Solver):
    """
    A solver implementing the Hungarian algorithm for minimum-cost bipartite matching.
    
    This implementation uses the Hungarian algorithm to find an optimal assignment
    that maximizes the total weight of the matching in a bipartite graph representation
    of the grid.
    
    Attributes:
    -----------
    grid: Grid
        The grid to solve
    pairs: list[tuple[tuple[int, int], tuple[int, int]]]
        A list of chosen pairs after running the algorithm
    cells: list[tuple[int, int]]
        A list of chosen cells after running the algorithm
    rules: str
        The matching rules to use ("original rules" or "new rules")
    """

    def __init__(self, grid: Grid):
        """
        Initializes the solver with a grid.

        Parameters:
        -----------
        grid: Grid
            The grid to solve
        """
        super().__init__(grid)
        self.rules = "original rules"

    def run(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Builds a bipartite cost matrix using only cells present in valid pairs.
        Applies the Hungarian algorithm to find optimal pairs.

        Returns:
        --------
        list[tuple[tuple[int, int], tuple[int, int]]]
            A list of pairs of cells, each represented as a tuple of tuples.

        Raises:
        -------
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

        # Update the cells list based on the chosen pairs
        self.cells = [cell for pair in self.pairs for cell in pair]
        
        return self.pairs

    def linear_sum_assignment(self, cost: np.ndarray, maximize: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the linear sum assignment problem.
        
        Parameters:
        -----------
        cost: np.ndarray
            The cost matrix of the bipartite graph.
        maximize: bool, optional
            Calculates a maximum weight matching if true. Default is False.
            
        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            An array of row indices and one of corresponding column indices giving the
            optimal assignment. The cost of the assignment can be computed as
            ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be sorted; in
            the case of a square cost matrix they will be equal to ``numpy.arange(cost_matrix.shape[0])``.
            
        Time Complexity:
        ---------------
        O(n³) where n is the dimension of the square cost matrix.
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

    def _find_short_augpath_while_body_inner_for(self, it: int, val: tuple) -> tuple:
        """
        Helper method for finding shortest augmenting paths in the Hungarian algorithm.
        
        Parameters:
        -----------
        it: int
            Current iteration index
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        tuple
            Updated algorithm state variables
        """
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

    def _find_short_augpath_while_body_tail(self, val: tuple) -> tuple:
        """
        Helper method for the tail part of processing in finding shortest augmenting paths.
        
        Parameters:
        -----------
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        tuple
            Updated algorithm state variables
        """
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

    def _find_short_augpath_while_body(self, val: tuple) -> tuple:
        """
        Main body of the algorithm for finding shortest augmenting paths.
        
        Parameters:
        -----------
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        tuple
            Updated algorithm state variables
        """
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

    def _find_short_augpath_while_cond(self, val: tuple) -> bool:
        """
        Condition check for continuing the search for shortest augmenting paths.
        
        Parameters:
        -----------
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        bool
            True if the search should continue, False otherwise
        """
        sink = val[-1]
        return sink == -1

    def _find_augmenting_path(self, cost: np.ndarray, u: np.ndarray, 
                             v: np.ndarray, path: np.ndarray, 
                             row4col: np.ndarray, current_row: int) -> tuple:
        """
        Find an augmenting path in the Hungarian algorithm.
        
        Parameters:
        -----------
        cost: np.ndarray
            Cost matrix
        u: np.ndarray
            Row potential vector
        v: np.ndarray
            Column potential vector
        path: np.ndarray
            Path array
        row4col: np.ndarray
            Row assignments for each column
        current_row: int
            Current row being processed
            
        Returns:
        --------
        tuple
            Sink node, minimum value, and other algorithm state variables
        """
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

    def _augment_previous_while_body(self, val: tuple) -> tuple:
        """
        Process for augmenting the previous matching along the found path.
        
        Parameters:
        -----------
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        tuple
            Updated algorithm state variables
        """
        path, sink, row4col, col4row, current_row, _ = val

        i = path[sink]
        row4col[sink] = i

        col4row[i], sink = sink, col4row[i]
        breakvar = i == current_row

        return path, sink, row4col, col4row, current_row, breakvar

    def _augment_previous_while_cond(self, val: tuple) -> bool:
        """
        Condition check for continuing the augmentation process.
        
        Parameters:
        -----------
        val: tuple
            Tuple containing algorithm state variables
            
        Returns:
        --------
        bool
            True if the augmentation should continue, False otherwise
        """
        breakvar = val[-1]
        return not breakvar

    def _lsa_body(self, cost: np.ndarray, u: np.ndarray, v: np.ndarray, 
                 path: np.ndarray, row4col: np.ndarray, 
                 col4row: np.ndarray, current_row: int) -> tuple:
        """
        Main body of the linear sum assignment algorithm (Hungarian algorithm).
        
        Parameters:
        -----------
        cost: np.ndarray
            Cost matrix
        u: np.ndarray
            Row potential vector
        v: np.ndarray
            Column potential vector
        path: np.ndarray
            Path array
        row4col: np.ndarray
            Row assignments for each column
        col4row: np.ndarray
            Column assignments for each row
        current_row: int
            Current row being processed
            
        Returns:
        --------
        tuple
            Updated algorithm state variables
        """
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
