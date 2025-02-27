"""
A module for the solver class and implementations of it.
"""

from grid import Grid, Cell


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


# adapter avec liste et, m*i + j ? si je veux enregistrer capacité ????


class SolverFulkerson(Solver):
    """
    Working ? I hope
    """

    def __init__(self, grid):
        """
        Initializes the solver with a grid and sets up the residual graph
        """
        super().__init__(grid)
        self.residual_graph = {}  # Dictionary to represent residual graph
        self.adjacency_graph_init()

    def adjacency_graph_init(self):
        """
        Initialize the residual graph for Ford-Fulkerson algorithm
        """
        # Initialize source and sink nodes
        self.residual_graph["source"] = {}
        self.residual_graph["sink"] = {}

        # Identify even and odd cells
        even_cells = []
        odd_cells = []
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                cell_id = f"cell_{i}_{j}"
                # Initialize each cell's entry in the residual graph
                self.residual_graph[cell_id] = {}

                # Sort cells into even and odd based on i+j parity
                if (i + j) % 2 == 0:
                    even_cells.append((i, j))
                    # Connect source to even cells with capacity 1
                    self.residual_graph["source"][cell_id] = 1
                else:
                    odd_cells.append((i, j))
                    # Connect odd cells to sink with capacity 1
                    self.residual_graph[cell_id]["sink"] = 1

        # Connect even cells to adjacent odd cells
        for i, j in even_cells:
            cell_id = f"cell_{i}_{j}"
            # Check all potential adjacent cells (up, down, left, right)
            adjacents = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

            for adj_i, adj_j in adjacents:
                # Check if valid cell and if it's an odd cell (different parity)
                if (
                    0 <= adj_i < self.grid.n
                    and 0 <= adj_j < self.grid.m
                    and (adj_i + adj_j) % 2 != 0
                ):
                    adj_cell_id = f"cell_{adj_i}_{adj_j}"

                    # Check if this pair is not forbidden - fixed format for forbidden pairs check
                    pair = [(i, j), (adj_i, adj_j)]
                    if not self.grid.is_pair_forbidden(pair):
                        # Add edge with capacity 1 (since we can match at most once)
                        self.residual_graph[cell_id][adj_cell_id] = 1
                    else:
                        print(f"forbidden {pair}")  # For debugging purposes

    def find_augmenting_path(self):
        """
        Find an augmenting path from source to sink in the residual graph using BFS
        Returns the path as a list of node IDs, or None if no path exists
        """
        # Queue for BFS
        queue = ["source"]
        # Keep track of visited nodes and their predecessors
        visited = {"source": None}

        while queue:
            current = queue.pop(0)

            # If we reached the sink, construct and return the path
            # TODO : Changer l'implémentation pour mettre une queue du module collections (dequeue)
            if current == "sink":
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                path.reverse()
                return path

            # Explore neighbors in residual graph
            if current in self.residual_graph:
                for neighbor, capacity in self.residual_graph[current].items():
                    if neighbor not in visited and capacity > 0:
                        visited[neighbor] = current
                        queue.append(neighbor)

        # No path found
        return None

    def ford_fulkerson(self):
        """
        Implements the Ford-Fulkerson algorithm to find maximum flow/matching
        """
        # Initialize flow to 0
        max_flow = 0

        # Find augmenting paths and update residual graph
        path = self.find_augmenting_path()
        while path:
            # Find bottleneck capacity
            min_capacity = float("inf")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                min_capacity = min(min_capacity, self.residual_graph[u][v])

            # Update residual capacities
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Decrease forward capacity
                self.residual_graph[u][v] -= min_capacity

                # Increase backward capacity (create if doesn't exist)
                if v not in self.residual_graph:
                    self.residual_graph[v] = {}
                if u not in self.residual_graph[v]:
                    self.residual_graph[v][u] = 0
                self.residual_graph[v][u] += min_capacity

            # Increase max flow by the bottleneck capacity
            max_flow += min_capacity

            # Find next augmenting path
            path = self.find_augmenting_path()
            print(path)

        return max_flow

    def run(self):
        """
        Run the solver and return the matching pairs
        """
        # Run Ford-Fulkerson algorithm
        max_flow = self.ford_fulkerson()
        print(f"Maximum flow: {max_flow}")
        print(self.residual_graph)

        print("break")
        # Extract matching from residual graph - improved to check backward edges
        matching_pairs = []
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i + j) % 2 == 0:  # Only check even cells
                    cell_id = f"cell_{i}_{j}"
                    if cell_id in self.residual_graph:
                        for adj_i in range(self.grid.n):
                            for adj_j in range(self.grid.m):
                                if (adj_i + adj_j) % 2 == 1:  # Only check odd cells
                                    adj_cell_id = f"cell_{adj_i}_{adj_j}"

                                    # Check if there's a backward edge with positive capacity
                                    # This indicates flow was sent through this edge
                                    if (
                                        adj_cell_id in self.residual_graph
                                        and cell_id in self.residual_graph[adj_cell_id]
                                        and self.residual_graph[adj_cell_id][cell_id]
                                        > 0
                                    ):
                                        matching_pairs.append(((i, j), (adj_i, adj_j)))
                                        # Add cells to self.cells for scoring
                                        self.cells.append((i, j))
                                        self.cells.append((adj_i, adj_j))

        self.pairs = matching_pairs
        return matching_pairs
