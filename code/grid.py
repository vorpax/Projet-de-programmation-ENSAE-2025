"""
Module qui contient la classe Grid et ses méthodes associées pour
représenter et manipuler une grille de jeu.
"""

import os
import matplotlib.pyplot as plt

# Matrices de compatibilité des couleurs pour les paires de cellules
# Une valeur de 1 indique que les couleurs sont compatibles, 0 sinon
# Indices des couleurs: 0=blanc, 1=rouge, 2=bleu, 3=vert, 4=noir
BLANC_COMBINAISON_OK = [1, 1, 1, 1, 0]  # Blanc compatible avec tout sauf noir
ROUGE_COMBINAISON_OK = [1, 1, 1, 0, 0]  # Rouge compatible avec blanc, rouge, bleu
BLEU_COMBINAISON_OK = [1, 1, 1, 0, 0]   # Bleu compatible avec blanc, rouge, bleu
VERT_COMBINAISON_OK = [1, 0, 0, 1, 0]   # Vert compatible avec blanc et vert
NOIR_COMBINAISON_OK = [0, 0, 0, 0, 0]   # Noir n'est compatible avec aucune couleur

MATRICE_COULEUR_OK = [
    BLANC_COMBINAISON_OK,
    ROUGE_COMBINAISON_OK,
    BLEU_COMBINAISON_OK,
    VERT_COMBINAISON_OK,
    NOIR_COMBINAISON_OK,
]


class Grid:
    """
    Classe représentant la grille de jeu.

    Attributs:
    -----------
    n: int
        Nombre de lignes dans la grille
    m: int
        Nombre de colonnes dans la grille
    color: list[list[int]]
        La couleur de chaque cellule: color[i][j] est la valeur de couleur de la cellule (i, j)
        Les lignes sont numérotées de 0 à n-1 et les colonnes de 0 à m-1.
    value: list[list[int]]
        La valeur de chaque cellule: value[i][j] est la valeur de la cellule (i, j)
        Les lignes sont numérotées de 0 à n-1 et les colonnes de 0 à m-1.
    colors_list: list[str]
        Correspondance entre la valeur numérique des couleurs et leur représentation
        0="w" (blanc), 1="r" (rouge), 2="b" (bleu), 3="g" (vert), 4="k" (noir)
    cells: list[list[Cell]]
        Liste 2D d'objets Cell, accessibles par leurs coordonnées cells[i][j]
    cells_list: list[Cell]
        Liste à plat de tous les objets Cell dans la grille
    """

    def __init__(
        self,
        n: int,
        m: int,
        color: list[list[int]] = None,
        value: list[list[int]] = None,
    ) -> None:
        """
        Initialise la grille de jeu.

        Paramètres:
        -----------
        n: int
            Nombre de lignes dans la grille
        m: int
            Nombre de colonnes dans la grille
        color: list[list[int]], optionnel
            Couleurs des cellules de la grille. Par défaut, toutes les cellules sont 
            initialisées à la couleur 0 (blanc).
        value: list[list[int]], optionnel
            Valeurs des cellules de la grille. Par défaut, toutes les cellules ont 
            une valeur de 1.
        """
        self.n = n
        self.m = m
        
        # Initialisation des couleurs (valeur par défaut: blanc)
        if not color:
            color = [[0 for j in range(m)] for i in range(n)]
        self.color = color
        
        # Initialisation des valeurs (valeur par défaut: 1)
        if not value:
            value = [[1 for j in range(m)] for i in range(n)]
        self.value = value
        
        # Correspondance codes couleur => représentation
        self.colors_list: list[str] = ["w", "r", "b", "g", "k"]
        
        # Les listes de cellules seront initialisées par cell_init()
        self.cells_list: list[Cell] = []
        self.cells: list[list[Cell]] = []

    def __str__(self) -> str:
        """
        Retourne une représentation textuelle de la grille montrant les couleurs et les valeurs.

        Retourne:
        --------
        str
            Une chaîne formatée représentant la grille
        """
        output = f"Grille de taille {self.n} x {self.m}. Couleurs:\n"
        for i in range(self.n):
            output += f"{[self.colors_list[self.color[i][j]] for j in range(self.m)]}\n"
        output += "Valeurs:\n"
        for i in range(self.n):
            output += f"{self.value[i]}\n"
        return output

    def __repr__(self) -> str:
        """
        Retourne une représentation concise de la grille avec son nombre de lignes et colonnes.

        Retourne:
        --------
        str
            Une chaîne au format "<grid.Grid: n=X, m=Y>"
        """
        return f"<grid.Grid: n={self.n}, m={self.m}>"

    def plot(self) -> None:
        """
        Affiche une représentation visuelle de la grille en utilisant matplotlib.

        Crée une visualisation colorée de la grille où:
        - Chaque cellule est colorée selon son attribut de couleur
        - La valeur numérique de la cellule est affichée au centre
        - Des lignes de grille sont tracées entre les cellules

        Retourne:
        --------
        None
        """
        ax = plt.subplots()[1]

        # Définition des couleurs RGB pour chaque type de cellule
        rgb_tab = [
            (255, 255, 255),  # Blanc
            (208, 0, 0),      # Rouge
            (68, 114, 196),   # Bleu
            (112, 173, 71),   # Vert
            (0, 0, 0),        # Noir
        ]

        # Création de la carte des couleurs
        color_map = []
        for i in range(self.n):
            color_map.append([])
            for j in range(self.m):
                color_map[i].append(rgb_tab[self.color[i][j]])
                plt.text(j, i, self.value[i][j], ha="center", va="center")
        
        # Configuration des paramètres d'affichage
        ax.tick_params(length=0, labelsize="large", pad=10)

        # Affichage de la grille colorée
        ax.matshow(color_map)
        
        # Ajout des lignes de grille
        plt.gca().set_xticks([x - 0.5 for x in range(1, self.m)], minor="true")
        plt.gca().set_yticks([x - 0.5 for x in range(1, self.m)], minor="true")
        ax.grid(visible=True, which="minor")
        
        plt.show()

    def is_forbidden(self, i: int, j: int) -> bool:
        """
        Vérifie si la cellule (i, j) est noire (interdite) ou non.

        Paramètres:
        -----------
        i: int
            L'indice de ligne de la cellule
        j: int
            L'indice de colonne de la cellule

        Retourne:
        --------
        bool
            True si la cellule est noire (interdite), False sinon

        Lève:
        -------
        IndexError
            Si les coordonnées de la cellule sont hors limites
        """
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            raise IndexError("Les coordonnées de la cellule sont hors limites")

        return self.get_coordinate_color(i, j) == "k"

    def is_pair_forbidden(self, pair: list[tuple[int, int]]) -> bool:
        """
        Vérifie si une paire de cellules est interdite selon les règles de compatibilité des couleurs.

        Paramètres:
        -----------
        pair: list[tuple[int, int]]
            Une paire de cellules représentée par une liste de deux tuples [(i1, j1), (i2, j2)]
            où (i1, j1) sont les coordonnées de la première cellule et
            (i2, j2) sont les coordonnées de la deuxième cellule

        Retourne:
        --------
        bool
            True si la paire est interdite, False sinon

        Lève:
        -------
        IndexError
            Si les coordonnées de l'une des cellules sont hors limites
            
        Complexité temporelle: O(1)
            Temps constant car implique seulement des accès directs aux indices et à la matrice.
        """
        # Vérification des limites pour la première cellule
        if (
            pair[0][0] < 0
            or pair[0][0] >= self.n
            or pair[0][1] < 0
            or pair[0][1] >= self.m
        ):
            raise IndexError("Les coordonnées de la première cellule sont hors limites")
        
        # Vérification des limites pour la deuxième cellule
        if (
            pair[1][0] < 0
            or pair[1][0] >= self.n
            or pair[1][1] < 0
            or pair[1][1] >= self.m
        ):
            raise IndexError("Les coordonnées de la deuxième cellule sont hors limites")

        # Récupération des couleurs des cellules
        couleur1 = self.color[pair[0][0]][pair[0][1]]
        couleur2 = self.color[pair[1][0]][pair[1][1]]
        
        # Vérification de la compatibilité des couleurs à l'aide de la matrice de compatibilité
        return not MATRICE_COULEUR_OK[couleur1][couleur2]

    def cost(self, pair: tuple[tuple[int]]) -> int:
        """
        Calcule le coût d'une paire de cellules.

        Paramètres:
        -----------
        pair: tuple[tuple[int]]
            Une paire au format ((i1, j1), (i2, j2))

        Retourne:
        -----------
        cost: int
            Le coût de la paire, défini comme la valeur absolue
            de la différence entre les valeurs des deux cellules
            
        Complexité temporelle: O(1)
            Opération en temps constant car implique seulement des accès
            directs aux valeurs et un calcul simple.
        """

        if self.is_pair_forbidden(pair):
            return 0  # Évite les erreurs pour les paires interdites

        valeur1 = self.get_coordinate_value(pair[0][0], pair[0][1])
        valeur2 = self.get_coordinate_value(pair[1][0], pair[1][1])

        return abs(valeur1 - valeur2)

    def all_pairs(self) -> list[list[tuple[int, int]]]:
        """
        Retourne une liste de toutes les paires de cellules qui peuvent être prises ensemble.

        Cette méthode parcourt la grille et identifie toutes les paires de cellules adjacentes
        qui sont compatibles selon les règles de couleur.

        Retourne:
        --------
        list[list[tuple[int, int]]]
            Une liste de paires valides, où chaque paire est représentée par une liste de deux tuples
            [(i1, j1), (i2, j2)], où (i1, j1) et (i2, j2) sont les coordonnées de deux
            cellules adjacentes qui peuvent être appariées.
            
        Complexité temporelle: O(n*m)
            Où n est le nombre de lignes et m le nombre de colonnes dans la grille.
        """
        liste_of_pairs = []
        # Parcours de toutes les cellules de la grille
        for i in range(self.n):
            for j in range(self.m):
                # Vérification des adjacences verticales (cellule en dessous)
                if i + 1 < self.n:
                    paire = [(i, j), (i + 1, j)]
                    if not self.is_pair_forbidden(paire):
                        liste_of_pairs.append(paire)
                # Vérification des adjacences horizontales (cellule à droite)
                if j + 1 < self.m:
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
        Initialise toutes les cellules de la grille.

        Cette méthode crée des objets Cell pour chaque position dans la grille et les stocke
        dans deux structures différentes :
        - self.cells : Une liste 2D où les cellules sont accessibles par leurs coordonnées
        - self.cells_list : Une liste à plat contenant toutes les cellules

        Retourne:
        --------
        None
        
        Complexité temporelle: O(n*m)
            Où n est le nombre de lignes et m le nombre de colonnes dans la grille.
        """
        # Initialisation de la liste 2D des cellules
        self.cells = []
        for i in range(self.n):
            ligne = []
            for j in range(self.m):
                ligne.append(Cell(i, j, self.color[i][j], self.value[i][j]))
            self.cells.append(ligne)

        # Création de la liste à plat des cellules
        self.cells_list = [
            self.cells[i][j] for i in range(self.n) for j in range(self.m)
        ]

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
            
        Time Complexity: O(n*m)
            Where n is the number of rows and m is the number of columns in the grid.
            The method reads and processes the file contents line by line.
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
