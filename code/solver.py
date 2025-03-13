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
    A solver implementing the Hungarian algorithm for maximum bipartite matching.
    """

    def __init__(self, grid):
        """
        Initializes the solver with a grid and sets up the adjacency dictionary.
        """
        super().__init__(grid)
        self.dict_adjacency = {}

    def adjacency_dict_init(self):
        """
        Initializes the adjacency dictionary for the Hungarian algorithm.
        """
        dict_test = {}
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                dict_test[f"cell_{i}_{j}"] = float("inf")

        # Déjà, par défaut on a infini

        for cell in self.grid.cells_list:
            adjacents = [
                (cell.i + 1, cell.j),
                (cell.i - 1, cell.j),
                (cell.i, cell.j + 1),
                (cell.i, cell.j - 1),
            ]
            self.dict_adjacency[f"cell_{cell.i}_{cell.j}"] = dict_test

            for adj_i, adj_j in adjacents:
                if 0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m:
                    if not self.grid.is_pair_forbidden(
                        ((cell.i, cell.j), (adj_i, adj_j))
                    ):
                        self.dict_adjacency[f"cell_{cell.i}_{cell.j}"][
                            f"cell_{adj_i}_{adj_j}"
                        ] = self.grid.cost(((cell.i, cell.j), (adj_i, adj_j)))
