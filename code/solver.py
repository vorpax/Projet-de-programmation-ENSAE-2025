"""
A module for the solver class and its implementations.

This module contains the base Solver class and various solver implementations
including SolverEmpty, SolverGreedy, and SolverFulkerson (which uses the
Ford-Fulkerson algorithm for maximum bipartite matching).
"""

from math import inf
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
        """
        remaining_cells_cost = sum(
            self.grid.get_coordinate_value(cell[0], cell[1])
            for cell in self.all_cells
            if cell not in self.cells
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


# adapter avec liste et, m*i + j ? si je veux enregistrer capacité ???? (enft non pas besoin)


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

            path = self.find_augmenting_path()  # And again and again

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
    A solver implementing the Hungarian algorithm for minimum weight bipartite matching.

    This solver models the grid as a bipartite graph where:
    - Even cells (where i+j is even) are in one set
    - Odd cells (where i+j is odd) are in the other set
    - The cost of a pair is the absolute difference between their values

    The Hungarian Algorithm then finds the minimum cost assignment (matching) between
    the two sets of cells.

    Attributes:
    -----------
    grid: Grid
        The grid to solve
    even_cells: list[tuple[int, int]]
        List of cells with even parity (i+j is even)
    odd_cells: list[tuple[int, int]]
        List of cells with odd parity (i+j is odd)
    cost_matrix: list[list[int]]
        The cost matrix for the Hungarian algorithm
    """

    def __init__(self, grid):
        """
        Initializes the solver with a grid and sets up the needed data structures.

        Parameters:
        -----------
        grid: Grid
            The grid to solve
        """
        super().__init__(grid)
        self.grid.cell_init()  # Ensure cells are initialized
        self.even_cells = []
        self.odd_cells = []
        self.cost_matrix = []
        self._initialize_cells()

    def _initialize_cells(self):
        """
        Separates cells into even and odd groups based on coordinate parity.
        """
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i + j) % 2 == 0:
                    self.even_cells.append((i, j))
                else:
                    self.odd_cells.append((i, j))

    def _create_cost_matrix(self):
        """
        Creates the cost matrix for the Hungarian algorithm.

        The cost matrix has dimensions len(even_cells) x len(odd_cells).
        If a pair (even_cell, odd_cell) is valid, its cost is the absolute difference
        of their values. Otherwise, the cost is infinity.

        Returns:
        --------
        list[list[int]]
            The cost matrix with rows for even cells and columns for odd cells
        """
        # Initialize with infinity (represents invalid pairs)
        cost_matrix = [
            [inf for _ in range(len(self.odd_cells))]
            for _ in range(len(self.even_cells))
        ]

        # Fill in valid pairs with their costs
        for i, even_cell in enumerate(self.even_cells):
            for j, odd_cell in enumerate(self.odd_cells):
                # Check if cells are adjacent and form a valid pair
                if (
                    abs(even_cell[0] - odd_cell[0]) == 1 and even_cell[1] == odd_cell[1]
                ) or (
                    even_cell[0] == odd_cell[0] and abs(even_cell[1] - odd_cell[1]) == 1
                ):
                    # Check if the pair is not forbidden
                    if not self.grid.is_pair_forbidden([even_cell, odd_cell]):
                        cost_matrix[i][j] = self.grid.cost((even_cell, odd_cell))

        self.cost_matrix = cost_matrix
        return cost_matrix

    def _find_min_value_in_row(self, row):
        """
        Find the minimum value in a row of the cost matrix.

        Parameters:
        -----------
        row: list[int]
            A row of the cost matrix

        Returns:
        --------
        int
            The minimum value in the row, or 0 if all values are infinity
        """
        min_val = inf
        for val in row:
            if val < min_val:
                min_val = val
        return min_val if min_val != inf else 0

    def _find_min_value_in_col(self, matrix, col_index):
        """
        Find the minimum value in a column of the cost matrix.

        Parameters:
        -----------
        matrix: list[list[int]]
            The cost matrix
        col_index: int
            The index of the column

        Returns:
        --------
        int
            The minimum value in the column, or 0 if all values are infinity
        """
        min_val = inf
        for row in matrix:
            if row[col_index] < min_val:
                min_val = row[col_index]
        return min_val if min_val != inf else 0

    def _is_assignment_complete(self, assignment):
        """
        Check if the assignment is complete (all rows have an assignment).

        Parameters:
        -----------
        assignment: list[int]
            The current assignment

        Returns:
        --------
        bool
            True if the assignment is complete, False otherwise
        """
        return all(val != -1 for val in assignment)

    def _find_augmenting_path(self, matrix, row_cover, col_cover, assignment):
        """
        Find an augmenting path in the reduced cost matrix.

        Parameters:
        -----------
        matrix: list[list[int]]
            The reduced cost matrix
        row_cover: list[bool]
            Boolean list indicating which rows are covered
        col_cover: list[bool]
            Boolean list indicating which columns are covered
        assignment: list[int]
            The current assignment (row -> column)

        Returns:
        --------
        tuple[list[int], list[bool], list[bool]]
            Updated assignment, row_cover, and col_cover
        """
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0

        # Cover rows that are all infinity
        for i in range(n):
            if not row_cover[i] and all(
                matrix[i][j] == inf for j in range(m) if not col_cover[j]
            ):
                row_cover[i] = True

        # Cover columns that are all infinity
        for j in range(m):
            if not col_cover[j] and all(
                matrix[i][j] == inf for i in range(n) if not row_cover[i]
            ):
                col_cover[j] = True

        # Step 1: Find a non-covered zero
        iteration_count = 0
        max_iterations = n * m * 2  # Safe upper bound

        while iteration_count < max_iterations:
            iteration_count += 1
            # Find a non-covered zero
            zero_found = False
            zero_row, zero_col = -1, -1

            for i in range(n):
                if row_cover[i]:
                    continue
                for j in range(m):
                    if not col_cover[j] and matrix[i][j] == 0:
                        zero_row, zero_col = i, j
                        zero_found = True
                        break
                if zero_found:
                    break

            # If no non-covered zero is found, find the minimum non-covered value
            if not zero_found:
                # Check if all rows are covered or all columns are covered
                if all(row_cover) or all(col_cover):
                    return assignment, row_cover, col_cover

                # Find the minimum non-covered value
                min_val = inf
                for i in range(n):
                    if row_cover[i]:
                        continue
                    for j in range(m):
                        if not col_cover[j] and matrix[i][j] < min_val:
                            min_val = matrix[i][j]

                # If min_val is still infinity, it means all remaining cells are infinity
                if min_val == inf:
                    # Cover all remaining rows and columns
                    for i in range(n):
                        row_cover[i] = True
                    for j in range(m):
                        col_cover[j] = True
                    return assignment, row_cover, col_cover

                # Add the minimum value to covered rows
                for i in range(n):
                    if row_cover[i]:
                        for j in range(m):
                            if matrix[i][j] != inf:
                                matrix[i][j] += min_val

                # Subtract the minimum value from non-covered columns
                for j in range(m):
                    if not col_cover[j]:
                        for i in range(n):
                            if matrix[i][j] != inf:
                                matrix[i][j] -= min_val

                continue

            # Mark the zero
            assignment[zero_row] = zero_col

            # Check if the column already has an assignment
            col_assigned = False
            for i in range(n):
                if i != zero_row and assignment[i] == zero_col:
                    col_assigned = True
                    row_cover[i] = False  # Uncover the row
                    col_cover[zero_col] = True  # Cover the column
                    break

            # If column is not assigned, we've found an augmenting path
            if not col_assigned:
                return assignment, row_cover, col_cover

            # Cover the row of the zero
            row_cover[zero_row] = True
        if iteration_count == max_iterations:
            print("Maximum iterations reached")

    def hungarian_algorithm(self):
        """
        Implements the Hungarian algorithm for minimum cost bipartite matching.

        The algorithm:
        1. Create the cost matrix
        2. Reduce rows and columns to get zeros
        3. Find a set of independent zeros that form a valid assignment
        4. If the assignment is complete, return it
        5. Otherwise, adjust the cost matrix and repeat

        Returns:
        --------
        list[tuple[tuple[int, int], tuple[int, int]]]
            A list of matched pairs with minimum total cost
        """
        cost_matrix = self._create_cost_matrix()
        n = len(cost_matrix)
        m = len(cost_matrix[0]) if n > 0 else 0

        if n == 0 or m == 0:
            return []

        # Step 1: Subtract row minima
        for i in range(n):
            min_val = self._find_min_value_in_row(cost_matrix[i])
            for j in range(m):
                if cost_matrix[i][j] != inf:
                    cost_matrix[i][j] -= min_val

        # Step 2: Subtract column minima
        for j in range(m):
            min_val = self._find_min_value_in_col(cost_matrix, j)
            for i in range(n):
                if cost_matrix[i][j] != inf:
                    cost_matrix[i][j] -= min_val

        # Step 3: Find an initial assignment
        assignment = [-1] * n  # Maps each row to a column (-1 means unassigned)

        # Try to assign zeros
        for i in range(n):
            for j in range(m):
                if cost_matrix[i][j] == 0 and assignment[i] == -1:
                    # Check if column j is not assigned yet
                    col_assigned = False
                    for k in range(i):
                        if assignment[k] == j:
                            col_assigned = True
                            break

                    if not col_assigned:
                        assignment[i] = j

        # Step 4: If the assignment is not complete, find augmenting paths
        while not self._is_assignment_complete(assignment):
            row_cover = [False] * n
            col_cover = [False] * m

            # Mark all rows with assignments
            for i in range(n):
                if assignment[i] != -1:
                    row_cover[i] = True

            # Find an augmenting path
            assignment, row_cover, col_cover = self._find_augmenting_path(
                cost_matrix, row_cover, col_cover, assignment
            )

        # Convert the assignment to pairs
        pairs = []
        for i, j in enumerate(assignment):
            if j != -1:
                even_cell = self.even_cells[i]
                odd_cell = self.odd_cells[j]
                pairs.append((even_cell, odd_cell))

        return pairs

    def run(self):
        """
        Runs the Hungarian algorithm solver and returns the matching pairs.

        Returns:
        --------
        list[tuple[tuple[int, int], tuple[int, int]]]
            A list of matched pairs, where each pair is represented as ((i1, j1), (i2, j2))
        """
        matching_pairs = self.hungarian_algorithm()

        # Update the solver's pairs and cells lists
        self.pairs = matching_pairs
        self.cells = []
        for pair in matching_pairs:
            self.cells.append(pair[0])
            self.cells.append(pair[1])

        print(f"Hungarian algorithm found {len(matching_pairs)} pairs")
        return matching_pairs
