"""
A module for the solver class and its implementations.

This module contains the base Solver class and various solver implementations
including SolverEmpty, SolverGreedy, and SolverFulkerson (which uses the
Ford-Fulkerson algorithm for maximum bipartite matching).
"""

from math import inf
from grid import Grid
# https://en.wikipedia.org/wiki/Hungarian_algorithm


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
    A solver implementing the Hungarian algorithm for maximum bipartite matching.
    """

    def __init__(self, grid):
        """
        Initializes the solver with a grid and sets up the adjacency dictionary.
        """
        super().__init__(grid)
        self.dict_adjacency = {}
        self.cost_matrix = self.adjacency_dict_init().copy()
        self.marked_cols = set()
        self.marked_rows = set()
        self.row_assignment = {}
        self.col_assignment = {}

    def cost_matrix_init(self):
        """
        Initializes the cost matrix for the Hungarian algorithm.
        """
        processed_cells = []
        cost_row = [inf] * self.grid.m * self.grid.n
        cost_matrix = [cost_row.copy() for _ in range(self.grid.n * self.grid.m)]
        # cost_matrix = (
        #     [[inf].copy() * self.grid.m * self.grid.n].copy()
        #     * self.grid.m
        #     * self.grid.n
        # )
        print(cost_matrix)
        print("brek")
        # cost_matrix = [cost_matrix.copy() for _ in range(self.grid.n)]
        total_size = self.grid.n * self.grid.m  # i * self.grid.m + j
        for k in range(total_size):
            i = k % (self.grid.m - 1)
            j = k // (self.grid.m - 1)

            adjacents = [
                (i + 1, j),
                (i - 1, j),
                (i, j + 1),
                (i, j - 1),
            ]
            for adj_i, adj_j in adjacents:
                p = adj_i * self.grid.m + adj_j

                if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                    if not self.grid.is_pair_forbidden(((i, j), (adj_i, adj_j))):
                        cost_matrix[k][p] = self.grid.cost(((i, j), (adj_i, adj_j)))
                        cost_matrix[p][k] = self.grid.cost(((i, j), (adj_i, adj_j)))
            # for p in range(total_size):
            #     q = p % (self.grid.m - 1)
            #     r = p // (self.grid.m - 1)
            #     if ((i, j), (q, r)) in processed_cells:
            #         print("chomage, je suis au chomage")

            #     processed_cells.append(((i, j), (q, r)))
            #     print(f"i = {i}, j = {j}, q={q}, r={r} p = {p}, k = {k}")
            #     if not self.grid.is_pair_forbidden(((i, j), (q, r))):
            #         cost_matrix[k][p] = self.grid.cost(((i, j), (q, r)))
            #         # cost_matrix[p][k] = self.grid.cost(((i, j), (q, r)))
            #     else:
            #         print("pair forbidden i= {i}, j={j}, q={q}, r={r}")

        return cost_matrix

    def adjacency_dict_init(self):
        """
        Initializes the adjacency dictionary for the Hungarian algorithm.
        """
        dict_test = {}
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                dict_test[f"cell_{i}_{j}"] = inf

        for i in range(self.grid.n):
            for j in range(self.grid.m):
                self.dict_adjacency[f"cell_{i}_{j}"] = dict_test.copy()

        # Déjà, par défaut on a infini
        for cell in self.grid.cells_list:
            adjacents = [
                (cell.i + 1, cell.j),
                (cell.i - 1, cell.j),
                (cell.i, cell.j + 1),
                (cell.i, cell.j - 1),
            ]

            for adj_i, adj_j in adjacents:
                if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                    if not self.grid.is_pair_forbidden(
                        ((cell.i, cell.j), (adj_i, adj_j))
                    ):
                        self.dict_adjacency[f"cell_{cell.i}_{cell.j}"][
                            f"cell_{adj_i}_{adj_j}"
                        ] = self.grid.cost(((cell.i, cell.j), (adj_i, adj_j)))
                        self.dict_adjacency[f"cell_{adj_i}_{adj_j}"][
                            f"cell_{cell.i}_{cell.j}"
                        ] = self.grid.cost(((cell.i, cell.j), (adj_i, adj_j)))
        return self.dict_adjacency

    def step1(self):
        """
        Subtract the minimum value from each row of the cost matrix.
        """
        cost_matrix = self.cost_matrix.copy()
        # Iterate directly through the dictionary keys
        for row_key in cost_matrix:
            row_values = list(cost_matrix[row_key].values())
            min_value = min(row_values) if min(row_values) != inf else 0
            # Subtract min value from each element in the row
            for col_key in cost_matrix[row_key]:
                cost_matrix[row_key][col_key] -= min_value

        self.cost_matrix = cost_matrix
        return cost_matrix

    def step2(self):
        """
        Subtract the minimum value from each column of the cost matrix.
        """
        cost_matrix = self.cost_matrix.copy()
        # Get all column keys (should be same as row keys in your case)
        col_keys = list(cost_matrix.keys())

        # For each column
        for col_key in col_keys:
            # Gather all values in this column
            column_values = [
                cost_matrix[row_key][col_key]
                for row_key in cost_matrix
                if col_key in cost_matrix[row_key]
            ]

            min_value = (
                min(column_values) if column_values and min(column_values) != inf else 0
            )

            # Subtract min value from each element in the column
            for row_key in cost_matrix:
                if col_key in cost_matrix[row_key]:
                    cost_matrix[row_key][col_key] -= min_value

        self.cost_matrix = cost_matrix
        return cost_matrix

    def step3(self):
        """
        Marks rows and columns to find minimum number of lines covering all zeros.
        """
        cost_matrix = self.cost_matrix.copy()

        # Step 1: Find a maximal assignment using a more thorough approach
        row_assignment = {}  # Maps row -> column
        col_assignment = {}  # Maps column -> row

        # Use Hungarian matching algorithm for initial assignment
        # (This is a simplified approach - a full implementation would use augmenting paths)
        for row_key in cost_matrix:
            for col_key, value in cost_matrix[row_key].items():
                if value == 0 and col_key not in col_assignment:
                    row_assignment[row_key] = col_key
                    col_assignment[col_key] = row_key
                    break

        # Step 2: Mark rows with no assignment
        unmarked_rows = set(cost_matrix.keys())
        marked_rows = {row for row in cost_matrix if row not in row_assignment}
        unmarked_rows -= marked_rows
        marked_cols = set()

        # Step 3-4: Iteratively mark columns and rows
        while True:
            # Mark columns with zeros in marked rows
            new_cols = set()
            for row in marked_rows:
                for col, value in cost_matrix[row].items():
                    if value == 0 and col not in marked_cols:
                        new_cols.add(col)

            if not new_cols:  # No new columns to mark
                break

            marked_cols.update(new_cols)

            # Mark rows with assignments in marked columns
            new_rows = set()
            for col in new_cols:
                if col in col_assignment and col_assignment[col] not in marked_rows:
                    new_rows.add(col_assignment[col])

            if not new_rows:  # No new rows to mark
                break

            marked_rows.update(new_rows)
            unmarked_rows -= new_rows

        # The minimum cover consists of unmarked rows and marked columns
        self.marked_rows = unmarked_rows  # Note: these are UNMARKED rows
        self.marked_cols = marked_cols
        self.row_assignment = row_assignment
        self.col_assignment = col_assignment
        return unmarked_rows, marked_cols

    def step4(self):
        """
        Checks if the minimum number of lines is equal to the size of the matrix
        """
        # The optimal solution is found when the number of lines equals the size of the matrix
        return len(self.marked_rows) + len(self.marked_cols) == len(self.cost_matrix)

    def step5(self):
        """
        Finds the smallest unmarked value in the cost matrix and updates the matrix.
        """
        cost_matrix = self.cost_matrix.copy()

        # Find the smallest unmarked value
        min_value = float("inf")
        for row_key in cost_matrix:
            if row_key not in self.marked_rows:  # Unmarked row
                for col_key, value in cost_matrix[row_key].items():
                    if (
                        col_key not in self.marked_cols and value < min_value
                    ):  # Unmarked column
                        min_value = value

        # Subtract from unmarked rows
        for row_key in cost_matrix:
            if row_key not in self.marked_rows:  # Unmarked row
                for col_key in cost_matrix[row_key]:
                    cost_matrix[row_key][col_key] -= min_value

        # Add to marked columns
        for row_key in cost_matrix:
            for col_key in cost_matrix[row_key]:
                if col_key in self.marked_cols:  # Marked column
                    cost_matrix[row_key][col_key] += min_value

        self.cost_matrix = cost_matrix
        return cost_matrix

    def find_maximum_zero_matching(self, cost_matrix):
        """Find maximum matching of zeros in cost matrix"""
        row_assignment = {}
        col_assignment = {}

        # Initial greedy assignment (same as before)
        for row_key in cost_matrix:
            for col_key, value in cost_matrix[row_key].items():
                if value == 0 and col_key not in col_assignment:
                    row_assignment[row_key] = col_key
                    col_assignment[col_key] = row_key
                    break

        # Try to improve the matching with alternating paths
        while True:
            # Find an unmatched row
            unmatched_row = None
            for row in cost_matrix:
                if row not in row_assignment:
                    unmatched_row = row
                    break

            if not unmatched_row:
                break  # All rows are matched

            # Try to find an augmenting path starting from this row
            path_found = self.find_augmenting_path_for_zeros(
                unmatched_row, cost_matrix, row_assignment, col_assignment
            )
            if not path_found:
                break  # No more augmenting paths

        return row_assignment, col_assignment

    def find_augmenting_path_for_zeros(
        self, start_row, cost_matrix, row_assignment, col_assignment
    ):
        """Find an augmenting path for zeros starting from a given row"""
        visited_rows = set()
        visited_cols = set()
        stack = [start_row]
        parent = {start_row: None}

        while stack:
            row = stack.pop()
            visited_rows.add(row)

            for col_key, value in cost_matrix[row].items():
                if value == 0 and col_key not in visited_cols:
                    visited_cols.add(col_key)
                    if col_key not in col_assignment:
                        # Found an augmenting path
                        self.augment_path(
                            col_key, row_assignment, col_assignment, parent
                        )
                        return True
                    else:
                        next_row = col_assignment[col_key]
                        if next_row not in visited_rows:
                            parent[next_row] = row
                            stack.append(next_row)

        return False

    def augment_path(self, col, row_assignment, col_assignment, parent):
        """Augment the matching along the found path"""
        while col is not None:
            row = parent[col]
            next_col = row_assignment.get(row)
            row_assignment[row] = col
            col_assignment[col] = row
            col = next_col

    def run(self):
        """
        Run the Hungarian algorithm to find the optimal matching
        """
        # Apply steps 1-5 until optimal solution is found
        self.step1()
        self.step2()

        # Maximum number of iterations to prevent infinite loops
        max_iterations = len(self.cost_matrix) ** 2
        iteration_count = 0

        while True:
            self.step3()
            if self.step4():
                print(
                    f"Hungarian algorithm converged after {iteration_count} iterations"
                )
                break

            if iteration_count >= max_iterations:
                print(
                    f"Warning: Hungarian algorithm reached maximum iterations ({max_iterations})"
                )
                break

            self.step5()
            iteration_count += 1

        # Extract the optimal matching
        matching_pairs = []
        cost_matrix = self.cost_matrix

        # Find cells with zero values in the final cost matrix
        used_cells = set()  # Keep track of cells already in a pair

        for row_key in cost_matrix:
            if row_key in used_cells:
                continue

            for col_key, value in cost_matrix[row_key].items():
                if col_key in used_cells:
                    continue

                if value == 0:
                    # Convert from 'cell_i_j' format to (i,j) coordinates
                    row_coords = tuple(map(int, row_key.split("_")[1:]))
                    col_coords = tuple(map(int, col_key.split("_")[1:]))

                    # Check if this is a valid pair (adjacent cells)
                    if not self.grid.is_pair_forbidden((row_coords, col_coords)):
                        matching_pairs.append((row_coords, col_coords))
                        # Mark these cells as used
                        used_cells.add(row_key)
                        used_cells.add(col_key)
                        break

        self.pairs = matching_pairs

        # Update the cells list
        self.cells = []
        for pair in self.pairs:
            self.cells.append(pair[0])
            self.cells.append(pair[1])

        return matching_pairs
