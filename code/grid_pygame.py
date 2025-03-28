"""
Grid Matching Game with PyGame Interface.

This module provides an interactive visual interface for the grid matching game
where players select adjacent cells to form pairs with the goal of minimizing
the total cost of the matching solution.
"""

import sys
import os
import pygame
from grid import Grid, Cell

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 80
PADDING = 5
HEADER_HEIGHT = 60
FOOTER_HEIGHT = 80
TEXT_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (255, 255, 0, 100)  # Yellow with alpha
SELECTED_COLOR = (0, 255, 0, 150)  # Green with alpha
MATCHED_COLOR = (0, 200, 255, 150)  # Blue with alpha
FONT_SIZE = 32
INFO_FONT_SIZE = 20

# RGB color mapping (matches the original grid.py colors)
COLOR_MAP = {
    "w": (255, 255, 255),  # White
    "r": (208, 0, 0),  # Red
    "b": (68, 114, 196),  # Blue
    "g": (112, 173, 71),  # Green
    "k": (0, 0, 0),  # Black
}


class GridGame:
    """
    Interactive grid matching game with pygame interface.

    Attributes:
        grid: The Grid object containing game data
        pairs: List of matched pairs
        score: Current game score
        selected_cell: Currently selected cell
        hovering_cell: Cell currently being hovered over
    """

    def __init__(self, grid_file):
        """
        Initialize the grid game with a grid loaded from a file.

        Args:
            grid_file: Path to the grid input file
        """
        # Load the grid
        self.grid = Grid.grid_from_file(grid_file, read_values=True)
        self.grid.cell_init()

        # Game state
        self.pairs = []
        self.selected_cell = None
        self.hovering_cell = None
        self.cells_in_pairs = set()  # Track cells that are already paired

        # Calculate window dimensions based on grid size
        self.width = self.grid.m * CELL_SIZE
        self.height = self.grid.n * CELL_SIZE + HEADER_HEIGHT + FOOTER_HEIGHT

        # Create pygame window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid Matching Game")

        # Create fonts
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.info_font = pygame.font.SysFont(None, INFO_FONT_SIZE)

    def draw_grid(self):
        """Draw the grid with cells and their values."""
        # Clear screen with light gray
        self.screen.fill((240, 240, 240))

        # Draw header with instructions
        pygame.draw.rect(
            self.screen, (220, 220, 220), (0, 0, self.width, HEADER_HEIGHT)
        )
        instructions = self.info_font.render(
            "Click adjacent cells to match pairs. Goal: Minimize total cost",
            True,
            TEXT_COLOR,
        )
        self.screen.blit(
            instructions, (10, HEADER_HEIGHT // 2 - instructions.get_height() // 2)
        )

        # Draw grid cells
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                # Cell position
                x = j * CELL_SIZE
                y = i * CELL_SIZE + HEADER_HEIGHT

                # Get cell color and value
                color_str = self.grid.get_coordinate_color(i, j)
                color_rgb = COLOR_MAP[color_str]
                value = self.grid.get_coordinate_value(i, j)

                # Draw cell rectangle
                pygame.draw.rect(
                    self.screen,
                    color_rgb,
                    (
                        x + PADDING,
                        y + PADDING,
                        CELL_SIZE - 2 * PADDING,
                        CELL_SIZE - 2 * PADDING,
                    ),
                )

                # Add border
                pygame.draw.rect(
                    self.screen,
                    (100, 100, 100),
                    (
                        x + PADDING,
                        y + PADDING,
                        CELL_SIZE - 2 * PADDING,
                        CELL_SIZE - 2 * PADDING,
                    ),
                    1,
                )

                # Draw cell value
                # Use black text for light colors, white for dark colors
                text_color = (
                    (255, 255, 255) if color_str in ["r", "b", "k"] else (0, 0, 0)
                )
                value_text = self.font.render(str(value), True, text_color)
                self.screen.blit(
                    value_text,
                    (
                        x + CELL_SIZE // 2 - value_text.get_width() // 2,
                        y + CELL_SIZE // 2 - value_text.get_height() // 2,
                    ),
                )

                # Highlight cell if it's being hovered over
                if self.hovering_cell == (i, j):
                    highlight = pygame.Surface(
                        (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                        pygame.SRCALPHA,
                    )
                    highlight.fill(HIGHLIGHT_COLOR)
                    self.screen.blit(highlight, (x + PADDING, y + PADDING))

                # Highlight selected cell
                if self.selected_cell == (i, j):
                    selected = pygame.Surface(
                        (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                        pygame.SRCALPHA,
                    )
                    selected.fill(SELECTED_COLOR)
                    self.screen.blit(selected, (x + PADDING, y + PADDING))

                # Highlight cells that are part of a pair
                if (i, j) in self.cells_in_pairs:
                    matched = pygame.Surface(
                        (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                        pygame.SRCALPHA,
                    )
                    matched.fill(MATCHED_COLOR)
                    self.screen.blit(matched, (x + PADDING, y + PADDING))

        # Draw footer with score and pairs info
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            (0, HEADER_HEIGHT + self.grid.n * CELL_SIZE, self.width, FOOTER_HEIGHT),
        )

        # Calculate and display score
        score = self.calculate_score()
        score_text = self.font.render(f"Score: {score}", True, TEXT_COLOR)
        pairs_text = self.info_font.render(
            f"Pairs: {len(self.pairs)}", True, TEXT_COLOR
        )

        # Display controls
        controls_text = self.info_font.render(
            "Press R to reset, ESC to quit", True, TEXT_COLOR
        )

        # Position and render texts
        self.screen.blit(score_text, (10, HEADER_HEIGHT + self.grid.n * CELL_SIZE + 10))
        self.screen.blit(pairs_text, (10, HEADER_HEIGHT + self.grid.n * CELL_SIZE + 45))
        self.screen.blit(
            controls_text,
            (
                self.width - controls_text.get_width() - 10,
                HEADER_HEIGHT + self.grid.n * CELL_SIZE + 30,
            ),
        )

        # Update display
        pygame.display.flip()

    def get_cell_at_pos(self, pos):
        """
        Get grid cell coordinates at the given screen position.

        Args:
            pos: (x, y) mouse position

        Returns:
            (i, j) grid coordinates or None if outside the grid
        """
        x, y = pos

        # Check if within grid bounds
        if (
            x < 0
            or x >= self.width
            or y < HEADER_HEIGHT
            or y >= HEADER_HEIGHT + self.grid.n * CELL_SIZE
        ):
            return None

        # Calculate grid coordinates
        j = x // CELL_SIZE
        i = (y - HEADER_HEIGHT) // CELL_SIZE

        # Validate against grid dimensions
        if 0 <= i < self.grid.n and 0 <= j < self.grid.m:
            return (i, j)

        return None

    def are_adjacent(self, cell1, cell2):
        """
        Check if two cells are adjacent.

        Args:
            cell1: (i1, j1) coordinates of first cell
            cell2: (i2, j2) coordinates of second cell

        Returns:
            True if the cells are adjacent, False otherwise
        """
        i1, j1 = cell1
        i2, j2 = cell2

        # Check if cells are horizontally or vertically adjacent
        return (abs(i1 - i2) == 1 and j1 == j2) or (abs(j1 - j2) == 1 and i1 == i2)

    def can_pair_cells(self, cell1, cell2):
        """
        Check if two cells can be paired according to game rules.

        Args:
            cell1: (i1, j1) coordinates of first cell
            cell2: (i2, j2) coordinates of second cell

        Returns:
            True if the cells can be paired, False otherwise
        """
        # Check if either cell is already paired
        if cell1 in self.cells_in_pairs or cell2 in self.cells_in_pairs:
            return False

        # Check if cells are adjacent
        if not self.are_adjacent(cell1, cell2):
            return False

        # Check if the pair is forbidden by color constraints
        return not self.grid.is_pair_forbidden([cell1, cell2])

    def add_pair(self, cell1, cell2):
        """
        Add a new pair to the game.

        Args:
            cell1: (i1, j1) coordinates of first cell
            cell2: (i2, j2) coordinates of second cell
        """
        if self.can_pair_cells(cell1, cell2):
            self.pairs.append([cell1, cell2])
            self.cells_in_pairs.add(cell1)
            self.cells_in_pairs.add(cell2)

    def calculate_score(self):
        """
        Calculate the current game score.

        Returns:
            The total score (cost of pairs + sum of unpaired cell values)
        """
        # Calculate cost of chosen pairs
        pair_cost = sum(self.grid.cost(pair) for pair in self.pairs)

        # Calculate value of unpaired cells (excluding black cells)
        unpaired_cost = 0
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) not in self.cells_in_pairs and self.grid.get_coordinate_color(
                    i, j
                ) != "k":
                    unpaired_cost += self.grid.get_coordinate_value(i, j)

        return pair_cost + unpaired_cost

    def reset_game(self):
        """Reset the game to initial state."""
        self.pairs = []
        self.selected_cell = None
        self.hovering_cell = None
        self.cells_in_pairs = set()

    def run(self):
        """Run the main game loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_game()

                elif event.type == pygame.MOUSEMOTION:
                    # Update hovering cell
                    self.hovering_cell = self.get_cell_at_pos(event.pos)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        clicked_cell = self.get_cell_at_pos(event.pos)

                        if clicked_cell:
                            # If no cell is selected, select this one
                            if self.selected_cell is None:
                                # Only select if not already paired
                                if clicked_cell not in self.cells_in_pairs:
                                    self.selected_cell = clicked_cell

                            # If a cell is already selected
                            else:
                                # If clicking the same cell, deselect it
                                if clicked_cell == self.selected_cell:
                                    self.selected_cell = None

                                # Try to pair with the selected cell
                                elif self.can_pair_cells(
                                    self.selected_cell, clicked_cell
                                ):
                                    self.add_pair(self.selected_cell, clicked_cell)
                                    self.selected_cell = None

                                # If can't pair, select the new cell if it's not already paired
                                elif clicked_cell not in self.cells_in_pairs:
                                    self.selected_cell = clicked_cell

            # Draw the updated grid
            self.draw_grid()

            # Cap the frame rate
            pygame.time.Clock().tick(60)

        pygame.quit()


def main():
    """Main function to start the game."""
    # Handle command line arguments for grid file
    data_path = "./input/"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = data_path + "grid28.in"

    # Make sure the file exists
    if not os.path.exists(file_name):
        print(f"Error: File {file_name} not found.")
        return

    # Create and run the game
    game = GridGame(file_name)
    game.run()


if __name__ == "__main__":
    main()
