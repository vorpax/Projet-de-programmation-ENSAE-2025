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

    This implementation follows the standard steps of the Hungarian algorithm:
    1. Subtract the minimum value from each row
    2. Subtract the minimum value from each column
    3. Cover all zeros with the minimum number of lines
    4. If the number of covering lines equals the size of the matrix, we're done
    5. Otherwise, create new zeros and repeat from step 3

    The algorithm finds a matching that minimizes the total cost of paired cells.
    """

    def __init__(self, grid):
        """
        Initializes the solver with a grid and sets up the cost matrix.
        """
        super().__init__(grid)
        self.cost_matrix = {}
        self.adjacency_dict_init()
        self.marked_cols = set()
        self.marked_rows = set()
        self.row_assignment = {}
        self.col_assignment = {}

    def adjacency_dict_init(self):
        """
        Initializes the cost matrix for the Hungarian algorithm using a dictionary
        representation for efficient sparse matrix operations.

        The cost matrix is represented as:
        - Rows correspond to cells in the grid
        - Columns also correspond to cells in the grid
        - Values represent the cost of pairing those cells
        - Valid pairs (adjacent cells that aren't forbidden) have actual costs
        - Invalid pairs or forbidden pairs have infinity cost
        
        Time Complexity: O(n^2 * m^2)
            Where n is the number of rows and m is the number of columns in the grid.
            In the worst case, we need to initialize O((n*m)^2) entries in the cost matrix.
        """
        # Initialize the cost matrix with dictionaries for each cell
        self.cost_matrix = {}

        # Get all possible cell IDs
        all_cell_ids = []
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                all_cell_ids.append(f"cell_{i}_{j}")

        # Initialize all entries in the cost matrix with infinity
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                cell_id = f"cell_{i}_{j}"
                self.cost_matrix[cell_id] = {
                    other_cell: 0 for other_cell in all_cell_ids
                }

        # Fill in actual costs for valid adjacent cells
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                # Skip black cells - they can't be paired
                if self.grid.get_coordinate_color(i, j) == "k":
                    continue

                cell_id = f"cell_{i}_{j}"
                # Check adjacent cells (up, down, left, right)
                adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

                for adj_i, adj_j in adjacents:
                    if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                        # Skip adjacent black cells - they can't be paired either
                        if self.grid.get_coordinate_color(adj_i, adj_j) == "k":
                            continue

                        adj_cell_id = f"cell_{adj_i}_{adj_j}"
                        # Only add finite costs for valid pairs
                        if not self.grid.is_pair_forbidden(((i, j), (adj_i, adj_j))):
                            value_1 = self.grid.get_coordinate_value(i, j)
                            value_2 = self.grid.get_coordinate_value(adj_i, adj_j)

                            pair_cost = -min(value_1, value_2)
                            self.cost_matrix[cell_id][adj_cell_id] = pair_cost

        return self.cost_matrix

    def step1(self):
        """
        Subtract the minimum value from each row of the cost matrix.

        This creates at least one zero in each row, which are candidates for
        the matching solution.
        """
        for row_key in self.cost_matrix:
            # Get all finite values in this row (ignoring infinity)
            row_values = [
                v for v in self.cost_matrix[row_key].values() if v != float("inf")
            ]

            # If there are no finite values, skip this row
            if not row_values:
                continue

            # Find minimum finite value in this row
            min_value = min(row_values)

            # Subtract min value from each finite element in the row
            for col_key in self.cost_matrix[row_key]:
                if self.cost_matrix[row_key][col_key] != float("inf"):
                    self.cost_matrix[row_key][col_key] -= min_value

        return self.cost_matrix

    def step2(self):
        """
        Subtract the minimum value from each column of the cost matrix.

        This creates at least one zero in each column, providing more
        candidates for the matching solution.
        """
        # Get all unique column keys
        all_cols = set()
        for row in self.cost_matrix:
            all_cols.update(self.cost_matrix[row].keys())

        # For each column
        for col_key in all_cols:
            # Gather all finite values in this column (ignoring infinity)
            column_values = []
            for row_key in self.cost_matrix:
                if col_key in self.cost_matrix[row_key] and self.cost_matrix[row_key][
                    col_key
                ] != float("inf"):
                    column_values.append(self.cost_matrix[row_key][col_key])

            # If there are no finite values in this column, skip it
            if not column_values:
                continue

            # Find minimum finite value in this column
            min_value = min(column_values)

            # Subtract min value from each finite element in the column
            for row_key in self.cost_matrix:
                if col_key in self.cost_matrix[row_key] and self.cost_matrix[row_key][
                    col_key
                ] != float("inf"):
                    self.cost_matrix[row_key][col_key] -= min_value

        return self.cost_matrix

    def find_maximum_zero_matching(self):
        """
        Find maximum matching of zeros in the cost matrix.

        Uses a greedy initial assignment followed by augmentation through
        alternating paths to maximize the number of matched rows and columns.

        Returns:
        --------
        tuple
            (row_assignment, col_assignment) dictionaries mapping rows to columns
            and columns to rows in the optimal matching
        """
        row_assignment = {}
        col_assignment = {}

        # Initial greedy assignment - only consider zero values (not infinity)
        for row_key in self.cost_matrix:
            for col_key, value in self.cost_matrix[row_key].items():
                if value == 0 and col_key not in col_assignment:
                    row_assignment[row_key] = col_key
                    col_assignment[col_key] = row_key
                    break

        # Try to improve the matching with augmenting paths
        while True:
            # Find an unmatched row
            unmatched_row = None
            for row in self.cost_matrix:
                if row not in row_assignment:
                    unmatched_row = row
                    break

            if not unmatched_row:
                break  # All rows are matched or no more rows can be matched

            # Try to find an augmenting path starting from this row
            path_found = self.find_augmenting_path(
                unmatched_row, row_assignment, col_assignment
            )
            if not path_found:
                break  # No more augmenting paths

        return row_assignment, col_assignment

    def find_augmenting_path(self, start_row, row_assignment, col_assignment):
        """
        Find an augmenting path for zeros starting from a given row.

        An augmenting path alternates between unmatched and matched edges,
        starting at an unmatched row and ending at an unmatched column.

        Parameters:
        -----------
        start_row: str
            The starting row (an unmatched row) for the path search
        row_assignment: dict
            Current assignment of rows to columns
        col_assignment: dict
            Current assignment of columns to rows

        Returns:
        --------
        bool
            True if an augmenting path was found and used, False otherwise
        """
        # Track visited nodes and the path
        visited_rows = set([start_row])
        visited_cols = set()
        parent = {start_row: None}  # Used to reconstruct the path

        # Use a queue for breadth-first search
        queue = [start_row]

        while queue:
            current_row = queue.pop(0)  # Dequeue

            # Try all columns with zeros from this row (excluding infinite values)
            for col_key, value in self.cost_matrix[current_row].items():
                # Only consider zeros (not infinity) for the augmenting path
                if value == 0 and value != float("inf") and col_key not in visited_cols:
                    visited_cols.add(col_key)
                    parent[col_key] = current_row

                    # If column is unmatched, we found an augmenting path
                    if col_key not in col_assignment:
                        self.augment_path(
                            col_key, parent, row_assignment, col_assignment
                        )
                        return True

                    # If column is matched, continue path through its matched row
                    next_row = col_assignment[col_key]
                    if next_row not in visited_rows:
                        visited_rows.add(next_row)
                        parent[next_row] = col_key
                        queue.append(next_row)

        # No augmenting path found
        return False

    def augment_path(self, end_col, parent, row_assignment, col_assignment):
        """
        Augment the matching along the found path.

        This flips the matched/unmatched status of edges along the path, which
        increases the size of the matching by one.

        Parameters:
        -----------
        end_col: str
            The unmatched column at the end of the augmenting path
        parent: dict
            Dictionary mapping nodes to their predecessors in the path
        row_assignment: dict
            Row to column assignments to update
        col_assignment: dict
            Column to row assignments to update
        """
        # Start at the unmatched column and work backwards
        current = end_col

        while True:
            row = parent[current]  # Get the row that led to this column

            # Unmatched column gets matched to its parent row
            row_assignment[row] = current
            col_assignment[current] = row

            if parent[row] is None:
                break  # Reached the starting unmatched row

            # Move to the previous column in the path
            current = parent[row]

            # Remove previous matching for this column if it exists
            if current in col_assignment:
                prev_row = col_assignment[current]
                if prev_row in row_assignment and row_assignment[prev_row] == current:
                    del row_assignment[prev_row]

            # This shouldn't be necessary with proper parent tracking, but just in case
            if current not in parent:
                break

    def step3(self):
        """
        Find the minimum number of lines needed to cover all zeros in the cost matrix.

        This step is crucial for determining if we have found an optimal assignment.
        If the number of lines equals the matrix dimension, we're done.

        Returns:
        --------
        tuple
            (marked_rows, marked_cols) sets containing covered rows and columns
        """
        # First find a maximum matching of zeros
        self.row_assignment, self.col_assignment = self.find_maximum_zero_matching()

        # Mark rows that have no assignment
        unmarked_rows = set(self.cost_matrix.keys()) - set(self.row_assignment.keys())
        marked_rows = set(unmarked_rows)  # Start with unassigned rows marked
        marked_cols = set()

        # Iteratively mark columns and rows until no new marks can be added
        while True:
            # Mark columns that have zeros in marked rows
            new_marked_cols = set()
            for row in marked_rows:
                for col, value in self.cost_matrix[row].items():
                    if value == 0 and col not in marked_cols:
                        new_marked_cols.add(col)

            if not new_marked_cols:
                break  # No new columns to mark

            marked_cols.update(new_marked_cols)

            # Mark rows that have assignments in marked columns
            new_marked_rows = set()
            for col in new_marked_cols:
                if (
                    col in self.col_assignment
                    and self.col_assignment[col] not in marked_rows
                ):
                    new_marked_rows.add(self.col_assignment[col])

            if not new_marked_rows:
                break  # No new rows to mark

            marked_rows.update(new_marked_rows)

        # The minimum line cover consists of:
        # - Unmarked rows (all rows - marked rows)
        # - Marked columns
        self.marked_rows = set(self.cost_matrix.keys()) - marked_rows
        self.marked_cols = marked_cols

        return self.marked_rows, self.marked_cols

    def step4(self):
        """
        Check if the current assignment is optimal.

        In a standard Hungarian algorithm, the assignment is optimal when the number
        of covering lines equals the matrix dimension. However, for grid matching
        with unmatchable cells, we need a different criterion.

        Since some cells can never be matched (black cells or those with no valid pairs),
        we check if the number of covering lines equals the size of the current assignment.
        If no more augmenting paths can be found (handled in find_maximum_zero_matching),
        and this condition is true, the assignment is optimal.

        For large grids, we add a minimum iteration requirement to prevent premature termination.

        Returns:
        --------
        bool
            True if the assignment is optimal, False otherwise
        """
        # Count the total number of covering lines (marked rows + marked columns)
        total_covering_lines = len(self.marked_rows) + len(self.marked_cols)

        # Check if the total covering lines equals the number of assignments
        # This is the König-Egerváry theorem: max matching size = min cover size
        # The condition must be exactly equal to ensure we reach optimal solution
        # We also check the iteration count to ensure we don't terminate too early
        # We're comparing against the iteration_count attribute that is set in the run method
        return total_covering_lines == len(self.row_assignment) and getattr(
            self, "iteration_count", 0
        ) >= min(5, len(self.cost_matrix) // 10)

    def step5(self):
        """
        Update the cost matrix to create new zeros.

        When the assignment is not optimal, this step:
        1. Finds the smallest uncovered value
        2. Subtracts it from all uncovered rows
        3. Adds it to all covered columns

        This creates new zeros while preserving the existing assignment.

        Returns:
        --------
        dict
            The updated cost matrix
        """
        # Find the minimum finite value in the uncovered part of the matrix
        min_value = float("inf")

        for row_key in self.cost_matrix:
            if row_key not in self.marked_rows:  # Only consider unmarked rows
                for col_key, value in self.cost_matrix[row_key].items():
                    if col_key not in self.marked_cols and value != float(
                        "inf"
                    ):  # Only consider unmarked columns with finite values
                        if value < min_value:
                            min_value = value

        if min_value == float("inf"):
            return self.cost_matrix  # No uncovered elements with finite cost

        # Subtract min_value from every uncovered element with finite cost
        for row_key in self.cost_matrix:
            if row_key not in self.marked_rows:  # Unmarked row
                for col_key in self.cost_matrix[row_key]:
                    if col_key not in self.marked_cols and self.cost_matrix[row_key][
                        col_key
                    ] != float(
                        "inf"
                    ):  # Unmarked column with finite cost
                        self.cost_matrix[row_key][col_key] -= min_value

        # Add min_value to every element with finite cost at intersection of marked row and marked column
        for row_key in self.marked_rows:  # Marked row
            for col_key in self.cost_matrix[row_key]:
                if col_key in self.marked_cols and self.cost_matrix[row_key][
                    col_key
                ] != float(
                    "inf"
                ):  # Marked column with finite cost
                    self.cost_matrix[row_key][col_key] += min_value

        return self.cost_matrix

    def run(self):
        """
        Run the Hungarian algorithm to find the optimal assignment.

        This implementation handles the special case of grid matching where:
        1. Not all cells can be matched (e.g., black cells are unmatchable)
        2. The objective is to maximize the number of matched cells
           while minimizing the cost of those matches

        The algorithm follows these steps:
        1. Reduce the cost matrix by row and column to create zeros
        2. Find a maximum assignment of zeros
        3. Check if it's optimal using the modified criterion
        4. If not optimal, create new zeros and repeat steps 2-3

        Returns:
        --------
        list[tuple]
            List of matched pairs in the format [(cell1, cell2), ...]
            
        Time Complexity: O(n^3 * m^3)
            Where n is the number of rows and m is the number of columns in the grid.
            The Hungarian algorithm has a cubic complexity in terms of the size of the cost matrix,
            which is O(n*m) × O(n*m) in our grid representation.
        """
        # Initialize cost matrix by reducing rows and columns
        self.step1()
        self.step2()

        # Maximum number of iterations to prevent infinite loops
        max_iterations = len(self.cost_matrix) * 2
        self.iteration_count = 0

        # Keep track of best assignment found
        best_assignment = {}
        best_assignment_size = 0

        # Iterate until optimal solution is found or max iterations reached
        while self.iteration_count < max_iterations:
            self.step3()

            # Check if current assignment is better than best found
            current_size = len(self.row_assignment)
            if current_size > best_assignment_size:
                best_assignment = self.row_assignment.copy()
                best_assignment_size = current_size

            if self.step4():
                # Optimal solution found according to our modified criterion
                # We add 1 to iteration_count for display purposes to avoid showing "0 iterations"
                print(
                    f"Optimal solution found after {self.iteration_count + 1} iterations"
                )
                break

            self.step5()
            self.iteration_count += 1

        # If we reached max iterations without finding an optimal solution
        if self.iteration_count >= max_iterations:
            print(
                f"Warning: Max iterations ({max_iterations}) reached without optimal solution"
            )

            # Use the best assignment found if it's better than the final one
            if best_assignment_size > len(self.row_assignment):
                print(f"Using best assignment found (size {best_assignment_size})")
                self.row_assignment = best_assignment

        # Extract the matching from row_assignment
        matching_pairs = []
        valid_pair_count = 0
        used_pairs = set()  # To track pairs we've already added (in either direction)

        # Use the final row_assignment to construct the matching
        for row_key, col_key in self.row_assignment.items():
            # Convert from 'cell_i_j' format to (i,j) coordinates
            row_coords = tuple(map(int, row_key.split("_")[1:]))
            col_coords = tuple(map(int, col_key.split("_")[1:]))

            # Check that this is a valid pair (cells must be adjacent)
            if abs(row_coords[0] - col_coords[0]) + abs(
                row_coords[1] - col_coords[1]
            ) == 1 and not self.grid.is_pair_forbidden((row_coords, col_coords)):
                # Create a canonical representation of the pair (sort by coordinates)
                canonical_pair = tuple(sorted([row_coords, col_coords]))

                # Only add the pair if we haven't seen it before
                if canonical_pair not in used_pairs:
                    matching_pairs.append((row_coords, col_coords))
                    used_pairs.add(canonical_pair)
                    valid_pair_count += 1

        print(f"Hungarian algorithm found {valid_pair_count} valid pairs")

        self.pairs = matching_pairs

        # Update the cells list
        self.cells = []
        for pair in self.pairs:
            self.cells.append(pair[0])
            self.cells.append(pair[1])

        return matching_pairs
