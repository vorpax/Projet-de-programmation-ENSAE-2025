"""
This is the grid module. It contains the Grid class and its associated methods.
"""

# Formation du binome: ensaeProg25Binome (en objet)
# Rendu : ensaeProg25Rendu (en objet)
# mail: ziyad.benomar@ensae.fr


BlancCombinaisonOk = [1, 1, 1, 1, 0]
RougeCombinaisonOk = [1, 1, 1, 0, 0]
BleuCombinaisonOk = [1, 1, 1, 0, 0]
VertCombinaisonOk = [1, 0, 0, 1, 0]
NoirCombinaisonOk = [0, 0, 0, 0, 0]

MatriceCouleurOk = [
    BlancCombinaisonOk,
    RougeCombinaisonOk,
    BleuCombinaisonOk,
    VertCombinaisonOk,
    NoirCombinaisonOk,
]


class Grid:
    """
    A class representing the grid.

    Attributes:
    -----------
    n: int
        Number of lines in the grid
    m: int
        Number of columns in the grid
    color: list[list[int]]
        The color of each grid cell: value[i][j] is the value in the cell (i, j), i.e., in the i-th line and j-th column.
        Note: lines are numbered 0..n-1 and columns are numbered 0..m-1.
    value: list[list[int]]
        The value of each grid cell: value[i][j] is the value in the cell (i, j), i.e., in the i-th line and j-th column.
        Note: lines are numbered 0..n-1 and columns are numbered 0..m-1.
    colors_list: list[char]
        The mapping between the value of self.color[i][j] and the corresponding color
    """

    def __init__(self, n, m, color=[], value=[]):
        """
        Initializes the grid.

        Parameters:
        -----------
        n: int
            Number of lines in the grid
        m: int
            Number of columns in the grid
        color: list[list[int]]
            The grid cells colors. Default is empty (then the grid is created with each cell having color 0, i.e., white).
        value: list[list[int]]
            The grid cells values. Default is empty (then the grid is created with each cell having value 1).

        The object created has an attribute colors_list: list[char], which is the mapping between the value of self.color[i][j] and the corresponding color
        """
        self.n = n
        self.m = m
        if not color:
            color = [[0 for j in range(m)] for i in range(n)]
        self.color = color
        if not value:
            value = [[1 for j in range(m)] for i in range(n)]
        self.value = value
        self.colors_list = ["w", "r", "b", "g", "k"]

    def __str__(self):
        """
        Prints the grid as text.
        """
        output = f"The grid is {self.n} x {self.m}. It has the following colors:\n"
        for i in range(self.n):
            output += f"{[self.colors_list[self.color[i][j]] for j in range(self.m)]}\n"
        output += "and the following values:\n"
        for i in range(self.n):
            output += f"{self.value[i]}\n"
        return output

    def __repr__(self):
        """
        Returns a representation of the grid with number of rows and columns.
        """
        return f"<grid.Grid: n={self.n}, m={self.m}>"

    def plot(self):
        """
        Plots a visual representation of the grid using matplotlib.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = colors.ListedColormap(self.colors_list)
        bounds = list(range(len(self.colors_list) + 1))
        norm = colors.BoundaryNorm(bounds, cmap.N)

        data = self.color
        ax.imshow(data, cmap=cmap, norm=norm)

        # Add value labels
        for i in range(self.n):
            for j in range(self.m):
                ax.text(
                    j,
                    i,
                    str(self.value[i][j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        # Customize grid appearance
        ax.grid(True, which="major", color="black", linewidth=2)
        # ax.set_xticks(np.arange(-0.5, self.m, 1))
        # ax.set_yticks(np.arange(-0.5, self.n, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.title("Grid Visualization")
        plt.tight_layout()
        plt.show()

    def is_forbidden(self, i, j):
        """
        Returns True is the cell (i, j) is black and False otherwise
        """

        return self.get_coordinate_color(i, j) == "k"

    def is_pair_forbidden(self, pair):
        """
        Returns True if the pair is forbidden and False otherwise
        A bit more complex and relevant than simply checking if one of the cells is black
        """
        Couleur1 = self.color[pair[0][0]][pair[0][1]]
        Couleur2 = self.color[pair[1][0]][pair[1][1]]
        return not MatriceCouleurOk[Couleur1][Couleur2]

    def cost(self, pair: tuple[tuple[int]]) -> int:
        """
        Returns the cost of a pair

        Parameters:
        -----------
        pair: tuple[tuple[int]]
            A pair in the format ((i1, j1), (i2, j2))

        Output:
        -----------
        cost: int
            the cost of the pair defined as the absolute value of the difference between their values
        """

        Valeur1 = self.get_coordinate_value(pair[0][0], pair[0][1])
        Valeur2 = self.get_coordinate_value(pair[1][0], pair[1][1])
        return abs(Valeur1 - Valeur2)

    def all_pairs(self):
        """
        Returns a list of all pairs of cells that can be taken together.

        Outputs a list of tuples of tuples [(c1, c2), (c1', c2'), ...] where each cell c1 etc. is itself a tuple (i, j)
        """

        ListOfPairs = []
        for i in range(self.n):
            for j in range(self.m):
                if i + 1 != self.n:
                    Paire = [[i, j], [i + 1, j]]
                    if not self.is_pair_forbidden(Paire):
                        ListOfPairs.append(Paire)
                if j + 1 != self.m:
                    Paire = ((i, j), (i, j + 1))
                    if not self.is_pair_forbidden(Paire):
                        ListOfPairs.append(Paire)

        return ListOfPairs

    def get_coordinate_color(self, i, j):
        """
        Retourne la couleur de la cellule (i, j) (sous forme de string et pas de chiffre)
        """

        Ligne = self.color[i]
        ColorIndex = Ligne[j]
        print(ColorIndex)
        return self.colors_list[ColorIndex]

    def get_coordinate_value(self, i, j):
        ligne = self.value[i]
        value_index = ligne[j]
        print(value_index)
        return value_index

    @classmethod
    def grid_from_file(cls, file_name, read_values=False):
        """
        Creates a grid object from class Grid, initialized with the information from the file file_name.

        Parameters:
        -----------
        file_name: str
            Name of the file to load. The file must be of the format:
            - first line contains "n m"
            - next n lines contain m integers that represent the colors of the corresponding cell
            - next n lines [optional] contain m integers that represent the values of the corresponding cell
        read_values: bool
            Indicates whether to read values after having read the colors. Requires that the file has 2n+1 lines

        Output:
        -------
        grid: Grid
            The grid
        """
        with open(file_name, "r", encoding="utf-8") as file:
            n, m = map(int, file.readline().split())
            color = [[] for i_line in range(n)]
            for i_line in range(n):
                line_color = list(map(int, file.readline().split()))
                if len(line_color) != m:
                    raise Exception("Format incorrect")
                for j in range(m):
                    if line_color[j] not in range(5):
                        raise Exception("Invalid color")
                color[i_line] = line_color

            if read_values:
                value = [[] for i_line in range(n)]
                for i_line in range(n):
                    line_value = list(map(int, file.readline().split()))
                    if len(line_value) != m:
                        raise Exception("Format incorrect")
                    value[i_line] = line_value
            else:
                value = []

            grid = Grid(n, m, color, value)
        return grid


class Cell:
    """
    A class representing a cell in the grid.

    Attributes:
    -----------
    i: int
        The line number of the cell
    j: int
        The column number of the cell
    color: int
        The color of the cell
    value: int
        The value of the cell
    """

    def __init__(self, i, j, color, value):
        """
        Initializes the cell.

        Parameters:
        -----------
        i: int
            The line number of the cell
        j: int
            The column number of the cell
        color: int
            The color of the cell
        value: int
            The value of the cell
        """
        self.i = i
        self.j = j
        self.color = color
        self.value = value

    def __str__(self):
        """
        Prints the cell as text.
        """
        return (
            f"Cell ({self.i}, {self.j}) has color {self.color} and value {self.value}"
        )

    def __repr__(self):
        """
        Returns a representation of the cell with its coordinates.
        """
        return f"<grid.Cell: i={self.i}, j={self.j}>"
