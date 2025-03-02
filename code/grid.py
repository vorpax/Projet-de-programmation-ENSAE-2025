"""
This is the grid module. It contains the Grid class and its associated methods.
"""

import os
import matplotlib.pyplot as plt

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
        Number of rows in the grid
    m: int
        Number of columns in the grid
    color: list[list[int]]
        The color of each grid cell: color[i][j] is the color value in the cell (i, j)
        , i.e., in the i-th row and j-th column.
        Note: rows are numbered 0..n-1 and columns are numbered 0..m-1.
    value: list[list[int]]
        The value of each grid cell: value[i][j] is the value in the cell (i, j)
        , i.e., in the i-th row and j-th column.
        Note: rows are numbered 0..n-1 and columns are numbered 0..m-1.
    colors_list: list[str]
        The mapping between the value of self.color[i][j] and the corresponding color
    cells: list[list[Cell]]
        A 2D list of Cell objects, accessible by coordinates cells[i][j]
    cells_list: list[Cell]
        A flattened list of all Cell objects in the grid
    """

    def __init__(
        self,
        n: int,
        m: int,
        color: list[list[int]] = None,
        value: list[list[int]] = None,
    ) -> None:
        """
        Initializes the grid.

        Parameters:
        -----------
        n: int
            Number of rows in the grid
        m: int
            Number of columns in the grid
        color: list[list[int]], optional
            The grid cells colors. Default is None
            (then the grid is created with each cell having color 0, i.e., white).
        value: list[list[int]], optional
            The grid cells values. Default is None
            (then the grid is created with each cell having value 1).

        The object created has an attribute colors_list: list[str],
        which is the mapping between the value of self.color[i][j] and the corresponding color
        """
        self.n = n
        self.m = m
        if not color:
            color = [[0 for j in range(m)] for i in range(n)]
        self.color = color
        if not value:
            value = [[1 for j in range(m)] for i in range(n)]
        self.value = value
        self.colors_list: list[str] = ["w", "r", "b", "g", "k"]
        self.cells_list: list[Cell] = []
        self.cells: list[list[Cell]] = []

    def __str__(self) -> str:
        """
        Returns a string representation of the grid showing colors and values.

        Returns:
        --------
        str
            A formatted string representation of the grid
        """
        output = f"The grid is {self.n} x {self.m}. It has the following colors:\n"
        for i in range(self.n):
            output += f"{[self.colors_list[self.color[i][j]] for j in range(self.m)]}\n"
        output += "and the following values:\n"
        for i in range(self.n):
            output += f"{self.value[i]}\n"
        return output

    def __repr__(self) -> str:
        """
        Returns a concise representation of the grid with number of rows and columns.

        Returns:
        --------
        str
            A string in the format "<grid.Grid: n=X, m=Y>"
        """
        return f"<grid.Grid: n={self.n}, m={self.m}>"

    def plot(self) -> None:
        """
        Plots a visual representation of the grid using matplotlib.

        Creates a colored grid visualization where:
        - Each cell is colored according to its color attribute
        - The cell's numerical value is displayed in the center
        - Grid lines are drawn between cells

        Returns:
        --------
        None
        """
        ax = plt.subplots()[1]

        rgb_tab = [
            (255, 255, 255),  # White
            (208, 0, 0),  # Red
            (68, 114, 196),  # Blue
            (112, 173, 71),  # Green
            (0, 0, 0),  # Black
        ]

        color_map = []
        for i in range(self.n):
            color_map.append([])
            for j in range(self.m):
                color_map[i].append(rgb_tab[self.color[i][j]])
                plt.text(j, i, self.value[i][j], ha="center", va="center")
        ax.tick_params(length=0, labelsize="large", pad=10)

        ax.matshow(color_map)
        plt.gca().set_xticks([x - 0.5 for x in range(1, self.m)], minor="true")
        plt.gca().set_yticks([x - 0.5 for x in range(1, self.m)], minor="true")
        ax.grid(visible=True, which="minor")
        plt.show()

    def is_forbidden(self, i: int, j: int) -> bool:
        """
        Returns True if the cell (i, j) is black and False otherwise.

        Parameters:
        -----------
        i: int
            The row index of the cell
        j: int
            The column index of the cell

        Returns:
        --------
        bool
            True if the cell is black (forbidden), False otherwise

        Raises:
        -------
        IndexError
            If the cell coordinates are out of bounds
        """
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            raise IndexError("Cell coordinates out of bounds")

        return self.get_coordinate_color(i, j) == "k"

    def is_pair_forbidden(self, pair: list[tuple[int, int]]) -> bool:
        """
        Returns True if the pair is forbidden and False otherwise.
        A bit more complex and relevant than simply checking if one of the cells is black.

        Parameters:
        -----------
        pair: list[tuple[int, int]]
            A pair of cells represented as a list of two tuples [(i1, j1), (i2, j2)]
            where (i1, j1) are the coordinates of the first cell and
            (i2, j2) are the coordinates of the second cell

        Returns:
        --------
        bool
            True if the pair is forbidden, False otherwise

        Raises:
        -------
        IndexError
            If either cell's coordinates are out of bounds
        """
        if (
            pair[0][0] < 0
            or pair[0][0] >= self.n
            or pair[0][1] < 0
            or pair[0][1] >= self.m
        ):
            raise IndexError("First cell coordinates out of bounds")
        if (
            pair[1][0] < 0
            or pair[1][0] >= self.n
            or pair[1][1] < 0
            or pair[1][1] >= self.m
        ):
            raise IndexError("Second cell coordinates out of bounds")

        couleur1 = self.color[pair[0][0]][pair[0][1]]
        couleur2 = self.color[pair[1][0]][pair[1][1]]
        return not MatriceCouleurOk[couleur1][couleur2]

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
            the cost of the pair defined as the absolute value
            of the difference between their values
        """

        if self.is_pair_forbidden(pair):
            return 0  # Essayer de prévenir d'éventuelles erreurs

        valeur1 = self.get_coordinate_value(pair[0][0], pair[0][1])
        valeur2 = self.get_coordinate_value(pair[1][0], pair[1][1])
        return abs(valeur1 - valeur2)

    def all_pairs(self) -> list[list[tuple[int, int]]]:
        """
        Returns a list of all pairs of cells that can be taken together.

        Returns:
        --------
        list[list[tuple[int, int]]]
            A list of valid cell pairs, where each pair is represented as a list of two tuples
            [(i1, j1), (i2, j2)], where (i1, j1) and (i2, j2) are the coordinates of two
            adjacent cells that can be paired together.
        """
        liste_of_pairs = []
        for i in range(self.n):
            for j in range(self.m):
                if i + 1 != self.n:
                    paire = [(i, j), (i + 1, j)]
                    if not self.is_pair_forbidden(paire):
                        liste_of_pairs.append(paire)
                if j + 1 != self.m:
                    paire = [(i, j), (i, j + 1)]
                    if not self.is_pair_forbidden(paire):
                        liste_of_pairs.append(paire)

        return liste_of_pairs

    def get_coordinate_color(self, i: int, j: int) -> str:
        """
        Returns the color of cell (i, j) as a string instead of a number.

        Parameters:
        -----------
        i: int
            The row index of the cell
        j: int
            The column index of the cell

        Returns:
        --------
        str
            The color of the cell as a string ('w', 'r', 'b', 'g', or 'k')
        """
        ligne = self.color[i]
        color_index = ligne[j]
        return self.colors_list[color_index]

    def get_coordinate_value(self, i: int, j: int) -> int:
        """
        Returns the value of cell (i, j).

        Parameters:
        -----------
        i: int
            The row index of the cell
        j: int
            The column index of the cell

        Returns:
        --------
        int
            The numerical value of the cell
        """
        ligne = self.value[i]
        value_index = ligne[j]
        return value_index

    def cell_init(self) -> None:
        """
        Initializes all the cells of the Grid.

        This method creates Cell objects for each position in the grid and stores them
        in two different structures:
        - self.cells: A 2D list where cells can be accessed by their coordinates
        - self.cells_list: A flattened list containing all cells

        Returns:
        --------
        None
        """
        for i in range(self.n):
            self.cells.append([])
            for j in range(self.m):
                self.cells[i].append(Cell(i, j, self.color[i][j], self.value[i][j]))

        self.cells_list = [
            Cell(i, j, self.color[i][j], self.value[i][j])
            for i in range(self.n)
            for j in range(self.m)
        ]

        # Vraiment pas besoin de DFS ou BFS.....

    @classmethod
    def grid_from_file(cls, file_name: str, read_values: bool = False) -> "Grid":
        """
        Creates a grid object from class Grid,
        initialized with the information from the file file_name.

        Parameters:
        -----------
        file_name: str
            Name of the file to load. The file must be of the format:
            - first line contains "n m"
            - next n lines contain m integers
            that represent the colors of the corresponding cell
            - next n lines [optional] contain m integers
            that represent the values of the corresponding cell
        read_values: bool
            Indicates whether to read values after having read the colors.
            Requires that the file has 2n+1 lines

        Returns:
        --------
        Grid
            The grid initialized from the file data

        Raises:
        -------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file format is incorrect or has invalid color values
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"The file {file_name} does not exist.")

        with open(file_name, "r", encoding="utf-8") as file:
            n, m = map(int, file.readline().split())
            color = [[] for i_line in range(n)]
            for i_line in range(n):
                line_color = list(map(int, file.readline().split()))
                if len(line_color) != m:
                    raise ValueError("Format incorrect")
                for j in range(m):
                    if line_color[j] not in range(5):
                        raise ValueError("Invalid color")
                color[i_line] = line_color

            if read_values:
                value = [[] for i_line in range(n)]
                for i_line in range(n):
                    line_value = list(map(int, file.readline().split()))
                    if len(line_value) != m:
                        raise ValueError("Format incorrect")
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

    def __init__(self, i: int, j: int, color: int, value: int):
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
        self.iseven = (i + j) % 2

    def __str__(self) -> str:
        """
        Returns a string representation of the cell including coordinates, color, and value.

        Returns:
        --------
        str
            A formatted string with cell information
        """
        return (
            f"Cell ({self.i}, {self.j}) has color {self.color} and value {self.value}"
        )

    def __repr__(self) -> str:
        """
        Returns a concise representation of the cell with its coordinates.

        Returns:
        --------
        str
            A string in the format "<grid.Cell: i=X, j=Y>"
        """
        return f"<grid.Cell: i={self.i}, j={self.j}>"
