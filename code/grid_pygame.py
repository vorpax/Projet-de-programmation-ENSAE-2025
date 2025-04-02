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
from solver import SolverGreedy, SolverFulkerson, SolverHungarian

# Initialize pygame
pygame.init()

# Constants
CELL_SIZE = 80
PADDING = 5
HEADER_HEIGHT = 80
FOOTER_HEIGHT = 100
MENU_HEIGHT = 500
MENU_WIDTH = 600
TEXT_COLOR = (20, 20, 20)
HIGHLIGHT_COLOR = (255, 255, 0, 100)  # Yellow with alpha
SELECTED_COLOR = (0, 255, 0, 150)  # Green with alpha
MATCHED_COLOR = (0, 200, 255, 150)  # Blue with alpha
PLAYER1_COLOR = (0, 200, 100, 150)  # Green with alpha for Player 1
PLAYER2_COLOR = (200, 0, 100, 150)  # Red with alpha for Player 2 / AI
FONT_SIZE = 32
INFO_FONT_SIZE = 20
TITLE_FONT_SIZE = 48
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER_COLOR = (120, 170, 220)
BUTTON_TEXT_COLOR = (255, 255, 255)
DROPDOWN_COLOR = (80, 130, 180)
DROPDOWN_HOVER_COLOR = (100, 150, 200)
DROPDOWN_BORDER_COLOR = (40, 90, 140)

# RGB color mapping (matches the original grid.py colors)
COLOR_MAP = {
    "w": (255, 255, 255),  # White
    "r": (208, 0, 0),  # Red
    "b": (68, 114, 196),  # Blue
    "g": (112, 173, 71),  # Green
    "k": (0, 0, 0),  # Black
}


class Button:
    """
    Button class for creating interactive buttons in pygame.
    
    Attributes:
        rect: The rectangle defining the button's position and size
        text: The text to display on the button
        color: The button's background color
        hover_color: The button's color when hovered
        font: The font to use for the button text
        callback: Function to call when button is clicked
    """
    
    def __init__(self, rect, text, font, callback=None):
        """
        Initialize a button.
        
        Args:
            rect: The rectangle defining the button's position and size (x, y, width, height)
            text: The text to display on the button
            font: The font to use for the button text
            callback: Function to call when button is clicked
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER_COLOR
        self.font = font
        self.callback = callback
        self.hovered = False
        
    def draw(self, screen):
        """Draw the button on the screen."""
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (50, 50, 50), self.rect, 2, border_radius=5)  # Border
        
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def update(self, mouse_pos):
        """Update button state based on mouse position."""
        self.hovered = self.rect.collidepoint(mouse_pos)
        
    def handle_event(self, event):
        """Handle mouse events on the button."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.callback:
                self.callback()
                return True
        return False


class Dropdown:
    """
    Dropdown menu for selecting options.
    
    Attributes:
        rect: The rectangle defining the dropdown's position and size
        options: List of options to display
        font: The font to use for the text
        callback: Function to call when an option is selected
        selected: Currently selected option
    """
    
    def __init__(self, rect, options, font, callback=None, default_option=0):
        """
        Initialize a dropdown menu.
        
        Args:
            rect: The rectangle defining the dropdown's position and size (x, y, width, height)
            options: List of options to display
            font: The font to use for the text
            callback: Function to call when an option is selected
            default_option: Index of the default selected option
        """
        self.rect = pygame.Rect(rect)
        self.options = options
        self.font = font
        self.callback = callback
        self.expanded = False
        self.selected_index = default_option if default_option < len(options) else 0
        self.hovered_index = -1
        self.option_height = rect[3]  # Height of each option
        self.z_index = 10  # Z-index for drawing order (higher values are drawn on top)
        
    def draw(self, screen):
        """Draw the dropdown on the screen."""
        # Create a surface for dropdown items when expanded
        if self.expanded:
            # Count visible options (excluding selected one)
            visible_options = len(self.options) - 1
            if visible_options > 0:
                # Create a background surface for all dropdown options
                dropdown_height = self.option_height * (visible_options + 1)
                dropdown_surface = pygame.Surface((self.rect.width, dropdown_height), pygame.SRCALPHA)
                dropdown_surface.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw dropdown options to the surface
                visible_count = 0
                for i, option in enumerate(self.options):
                    if i == self.selected_index:
                        continue  # Skip already selected option
                    
                    # Calculate option position
                    option_rect = pygame.Rect(
                        0,  # Relative to surface
                        self.option_height * visible_count,
                        self.rect.width,
                        self.option_height
                    )
                    visible_count += 1
                    
                    # Highlight hovered option
                    if i == self.hovered_index:
                        pygame.draw.rect(dropdown_surface, DROPDOWN_HOVER_COLOR, option_rect, border_radius=5)
                    else:
                        pygame.draw.rect(dropdown_surface, DROPDOWN_COLOR, option_rect, border_radius=5)
                        
                    pygame.draw.rect(dropdown_surface, DROPDOWN_BORDER_COLOR, option_rect, 1, border_radius=5)
                    
                    option_text = self.font.render(option, True, BUTTON_TEXT_COLOR)
                    option_text_rect = option_text.get_rect(midleft=(option_rect.x + 10, option_rect.centery))
                    dropdown_surface.blit(option_text, option_text_rect)
                
                # Blit the entire dropdown surface to the screen
                screen.blit(dropdown_surface, (self.rect.x, self.rect.y + self.option_height))
        
        # Draw main dropdown button (always on top)
        pygame.draw.rect(screen, DROPDOWN_COLOR, self.rect, border_radius=5)
        pygame.draw.rect(screen, DROPDOWN_BORDER_COLOR, self.rect, 2, border_radius=5)
        
        # Draw selected option
        if self.options:
            text = self.options[self.selected_index]
            text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
            text_rect = text_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
            screen.blit(text_surf, text_rect)
            
            # Draw dropdown arrow
            arrow_points = [
                (self.rect.right - 20, self.rect.centery - 5),
                (self.rect.right - 10, self.rect.centery - 5),
                (self.rect.right - 15, self.rect.centery + 5)
            ]
            pygame.draw.polygon(screen, BUTTON_TEXT_COLOR, arrow_points)
    
    def update(self, mouse_pos):
        """Update dropdown state based on mouse position."""
        if self.expanded:
            # Check if hovering over an option
            self.hovered_index = -1
            visible_count = 0
            for i, option in enumerate(self.options):
                if i == self.selected_index:
                    continue  # Skip already selected option
                    
                # Calculate option position
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + self.option_height * (visible_count + 1),
                    self.rect.width,
                    self.option_height
                )
                visible_count += 1
                
                if option_rect.collidepoint(mouse_pos):
                    self.hovered_index = i
                    break
        
        # Always check if hovering over the main dropdown button
        self.main_hovered = self.rect.collidepoint(mouse_pos)
    
    def handle_event(self, event):
        """Handle mouse events on the dropdown."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return True
                
            if self.expanded:
                # Calculate visible options positions to check for clicks
                visible_count = 0
                for i, option in enumerate(self.options):
                    if i == self.selected_index:
                        continue  # Skip already selected option
                        
                    # Calculate option position
                    option_rect = pygame.Rect(
                        self.rect.x,
                        self.rect.y + self.option_height * (visible_count + 1),
                        self.rect.width,
                        self.option_height
                    )
                    visible_count += 1
                    
                    if option_rect.collidepoint(event.pos):
                        self.selected_index = i
                        self.expanded = False
                        if self.callback:
                            self.callback(self.options[self.selected_index])
                        return True
                
        # Close dropdown if clicked elsewhere
        if event.type == pygame.MOUSEBUTTONDOWN and self.expanded:
            self.expanded = False
            
        return False
        
    def get_selected(self):
        """Get the currently selected option."""
        return self.options[self.selected_index]


class GameResult:
    """
    Class to store and display game results.
    
    Attributes:
        player1_score: Score for player 1
        player2_score: Score for player 2 or AI
        player1_pairs: Number of pairs made by player 1
        player2_pairs: Number of pairs made by player 2 or AI
        winner: The winner of the game ("player1", "player2", or "tie")
        game_mode: The game mode that was played
        ai_algorithm: The AI algorithm used (if applicable)
    """
    
    def __init__(self, player1_score, player2_score, player1_pairs, player2_pairs, 
                 game_mode, ai_algorithm=None):
        """Initialize game result with scores and game mode details."""
        self.player1_score = player1_score
        self.player2_score = player2_score
        self.player1_pairs = player1_pairs
        self.player2_pairs = player2_pairs
        self.game_mode = game_mode
        self.ai_algorithm = ai_algorithm
        
        # Determine winner
        if player1_score < player2_score:  # Lower score wins
            self.winner = "player1"
        elif player2_score < player1_score:
            self.winner = "player2"
        else:
            self.winner = "tie"
    
    def draw(self, screen, font, info_font, title_font):
        """Draw the game result screen."""
        # Clear the screen
        screen.fill((240, 240, 240))
        
        # Draw title
        title = title_font.render("Game Finished", True, TEXT_COLOR)
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 30))
        
        # Draw winner
        if self.winner == "tie":
            winner_text = "It's a tie!"
        elif self.winner == "player1":
            winner_text = "Player 1 wins!" if self.game_mode == "vs_player" else "You win!"
        else:
            winner_text = "Player 2 wins!" if self.game_mode == "vs_player" else f"AI ({self.ai_algorithm}) wins!"
            
        winner_surf = font.render(winner_text, True, 
                                (0, 128, 0) if self.winner == "player1" else 
                                (200, 0, 0) if self.winner == "player2" else TEXT_COLOR)
        screen.blit(winner_surf, (screen.get_width() // 2 - winner_surf.get_width() // 2, 100))
        
        # Draw score breakdown
        y_pos = 160
        score_texts = [
            f"Player 1 score: {self.player1_score}" if self.game_mode == "vs_player" else f"Your score: {self.player1_score}",
            f"Player 1 pairs: {self.player1_pairs}" if self.game_mode == "vs_player" else f"Your pairs: {self.player1_pairs}",
            f"Player 2 score: {self.player2_score}" if self.game_mode == "vs_player" else f"AI score: {self.player2_score}",
            f"Player 2 pairs: {self.player2_pairs}" if self.game_mode == "vs_player" else f"AI pairs: {self.player2_pairs}",
        ]
        
        for text in score_texts:
            score_surf = info_font.render(text, True, TEXT_COLOR)
            screen.blit(score_surf, (screen.get_width() // 2 - score_surf.get_width() // 2, y_pos))
            y_pos += 30
        
        # Draw return to menu instruction
        instructions = info_font.render("Press M to return to menu or ESC to quit", True, TEXT_COLOR)
        screen.blit(instructions, (screen.get_width() // 2 - instructions.get_width() // 2, y_pos + 40))
        
        pygame.display.flip()


class GridGame:
    """
    Interactive grid matching game with pygame interface.

    Attributes:
        grid: The Grid object containing game data
        pairs: List of matched pairs (each pair includes who created it)
        player1_pairs: List of pairs made by player 1
        player2_pairs: List of pairs made by player 2 or AI
        score: Current game score
        selected_cell: Currently selected cell
        hovering_cell: Cell currently being hovered over
        game_mode: Current game mode ("solo", "vs_player", or "vs_ai")
        ai_algorithm: AI algorithm type when in vs_ai mode
        current_player: Current player (1 or 2)
    """

    def __init__(self, grid_file=None):
        """
        Initialize the grid game with a grid loaded from a file.

        Args:
            grid_file: Path to the grid input file (can be None for menu-only init)
        """
        # Get available grid files
        self.input_dir = "./input/"
        self.available_grids = self.get_available_grids()
        
        # Game state
        self.grid = None
        self.grid_file = grid_file
        self.pairs = []
        self.player1_pairs = []  # Pairs made by player 1
        self.player2_pairs = []  # Pairs made by player 2 or AI
        self.selected_cell = None
        self.hovering_cell = None
        self.cells_in_pairs = set()  # Track cells that are already paired (for solo mode)
        self.player1_cells = set()  # Track cells that are paired by player 1
        self.player2_cells = set()  # Track cells that are paired by player 2 or AI
        self.show_menu = True  # Start with menu screen
        self.show_results = False  # Show results screen at game end
        self.game_mode = None  # "solo", "vs_player", or "vs_ai"
        self.ai_algorithm = None  # "greedy", "fulkerson", or "hungarian"
        self.game_result = None  # To store game results
        
        # Player state
        self.current_player = 1  # Player 1 starts
        self.game_over = False  # Track if game is over
        
        # Create pygame window for menu initially
        self.menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
        pygame.display.set_caption("Grid Matching Game")

        # Create fonts
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.info_font = pygame.font.SysFont(None, INFO_FONT_SIZE)
        self.title_font = pygame.font.SysFont(None, TITLE_FONT_SIZE)
        
        # Create menu buttons
        self.create_menu_buttons()
        
        # If grid file provided, load it
        if grid_file:
            self.load_grid(grid_file)
            
    def get_available_grids(self):
        """Get list of available grid files from input directory."""
        grids = []
        for file in os.listdir(self.input_dir):
            if file.endswith('.in'):
                grids.append(file)
        return sorted(grids)
    
    def load_grid(self, grid_file):
        """Load a grid from a file."""
        if not os.path.exists(grid_file):
            print(f"Error: File {grid_file} not found.")
            return False
            
        try:
            self.grid = Grid.grid_from_file(grid_file, read_values=True)
            self.grid.cell_init()
            self.grid_file = grid_file
            
            # Calculate window dimensions based on grid size
            self.width = max(self.grid.m * CELL_SIZE, 600)  # Ensure minimum width
            self.height = self.grid.n * CELL_SIZE + HEADER_HEIGHT + FOOTER_HEIGHT
            
            return True
        except Exception as e:
            print(f"Error loading grid: {e}")
            return False

    def create_menu_buttons(self):
        """Create buttons for the menu screen."""
        button_width = 200
        button_height = 40
        button_spacing = 20
        dropdown_width = 300
        
        # Grid selection dropdown
        self.grid_dropdown = Dropdown(
            (MENU_WIDTH // 2 - dropdown_width // 2, 120, dropdown_width, button_height),
            self.available_grids,
            self.font,
            self.select_grid
        )
        
        # Game mode buttons
        self.solo_button = Button(
            (MENU_WIDTH // 2 - button_width // 2, 190, button_width, button_height),
            "Solo Game",
            self.font,
            lambda: self.start_game("solo")
        )
        
        self.vs_player_button = Button(
            (MENU_WIDTH // 2 - button_width // 2, 190 + button_height + button_spacing, button_width, button_height),
            "VS Player",
            self.font,
            lambda: self.start_game("vs_player")
        )
        
        # AI game buttons
        self.greedy_button = Button(
            (MENU_WIDTH // 2 - button_width // 2, 190 + 2 * (button_height + button_spacing), button_width, button_height),
            "VS Greedy AI",
            self.font,
            lambda: self.start_game("vs_ai", "greedy")
        )
        
        self.fulkerson_button = Button(
            (MENU_WIDTH // 2 - button_width // 2, 190 + 3 * (button_height + button_spacing), button_width, button_height),
            "VS Fulkerson AI",
            self.font,
            lambda: self.start_game("vs_ai", "fulkerson")
        )
        
        self.hungarian_button = Button(
            (MENU_WIDTH // 2 - button_width // 2, 190 + 4 * (button_height + button_spacing), button_width, button_height),
            "VS Hungarian AI",
            self.font,
            lambda: self.start_game("vs_ai", "hungarian")
        )
        
        self.menu_buttons = [
            self.solo_button, 
            self.vs_player_button,
            self.greedy_button, 
            self.fulkerson_button, 
            self.hungarian_button
        ]
    
    def select_grid(self, grid_name):
        """Handle grid selection from dropdown."""
        self.load_grid(os.path.join(self.input_dir, grid_name))
    
    def draw_menu(self):
        """Draw the menu screen."""
        # Clear screen
        self.menu_screen.fill((240, 240, 240))
        
        # Draw title
        title = self.title_font.render("Grid Matching Game", True, TEXT_COLOR)
        self.menu_screen.blit(title, (MENU_WIDTH // 2 - title.get_width() // 2, 20))
        
        # Draw game mode label
        mode_label = self.font.render("Select Game Mode:", True, TEXT_COLOR)
        self.menu_screen.blit(mode_label, (MENU_WIDTH // 2 - mode_label.get_width() // 2, 170))
        
        # Draw grid selection label
        grid_label = self.font.render("Select Grid:", True, TEXT_COLOR)
        self.menu_screen.blit(grid_label, (MENU_WIDTH // 2 - grid_label.get_width() // 2, 80))
        
        # Draw buttons - lower z-index
        for button in self.menu_buttons:
            button.draw(self.menu_screen)
            
        # Draw dropdown after buttons so it appears on top - higher z-index
        self.grid_dropdown.draw(self.menu_screen)
        
        # Update display
        pygame.display.flip()
    
    def get_player_cells(self, player_num):
        """
        Get the set of cells that are in pairs made by a specific player.
        
        Args:
            player_num: 1 for player 1, 2 for player 2 or AI
            
        Returns:
            Set of (i, j) coordinates of cells that are in pairs made by the player
        """
        return self.player1_cells if player_num == 1 else self.player2_cells
    
    def start_game(self, mode, algorithm=None):
        """
        Start a new game with the selected mode and algorithm.
        
        Args:
            mode: "solo", "vs_player", or "vs_ai"
            algorithm: "greedy", "fulkerson", or "hungarian" (only used in vs_ai mode)
        """
        # Make sure a grid is loaded
        if not self.grid:
            # If no grid is selected, use default
            if not self.load_grid(os.path.join(self.input_dir, self.available_grids[0])):
                return  # Failed to load grid
        
        # Set up game state
        self.show_menu = False
        self.show_results = False
        self.game_mode = mode
        self.ai_algorithm = algorithm
        self.game_over = False
        
        # Reset player state
        self.current_player = 1  # Player 1 starts
        self.pairs = []
        self.player1_pairs = []
        self.player2_pairs = []
        self.selected_cell = None
        self.hovering_cell = None
        self.cells_in_pairs = set()
        self.player1_cells = set()
        self.player2_cells = set()
        
        # Create game screen (larger size)
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        print(f"Game started in {mode} mode" + (f" with {algorithm} AI" if algorithm else ""))

    def draw_grid(self):
        """Draw the grid with cells and their values."""
        # Clear screen with light gray
        self.screen.fill((240, 240, 240))

        # Draw header with instructions and mode
        pygame.draw.rect(
            self.screen, (220, 220, 220), (0, 0, self.width, HEADER_HEIGHT)
        )
        
        # Display game mode
        if self.game_mode == "solo":
            mode_text = "Solo Mode"
        elif self.game_mode == "vs_player":
            mode_text = "VS Player Mode"
        else:
            mode_text = f"VS {self.ai_algorithm.capitalize()} AI"
        
        mode_info = self.font.render(mode_text, True, TEXT_COLOR)
        
        # Display current player
        if self.game_mode == "solo":
            player_text = ""
        elif self.game_mode == "vs_player":
            player_text = f"Player {self.current_player}'s Turn"
        else:
            if self.current_player == 1:
                player_text = "Your Turn"
            else:
                player_text = f"AI's Turn ({self.ai_algorithm})"
        
        player_info = self.info_font.render(player_text, True, 
                                (0, 128, 0) if self.current_player == 1 else (208, 0, 0))
        
        # Draw instructions
        instructions = self.info_font.render(
            "Click adjacent cells to match pairs. Goal: Minimize total cost",
            True,
            TEXT_COLOR,
        )
        
        # Position and blit text elements
        self.screen.blit(mode_info, (10, 10))
        self.screen.blit(player_info, (self.width - player_info.get_width() - 10, 10))
        self.screen.blit(
            instructions, (10, HEADER_HEIGHT // 2 - instructions.get_height() // 2 + 10)
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

                # Highlight cells that are part of a pair with player colors
                cell_highlight = None
                
                # In solo mode
                if self.game_mode == "solo" and (i, j) in self.cells_in_pairs:
                    cell_highlight = pygame.Surface(
                        (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                        pygame.SRCALPHA,
                    )
                    cell_highlight.fill(MATCHED_COLOR)
                    
                # In multiplayer modes with separate player highlights
                elif self.game_mode in ["vs_player", "vs_ai"]:
                    if (i, j) in self.player1_cells:
                        cell_highlight = pygame.Surface(
                            (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                            pygame.SRCALPHA,
                        )
                        cell_highlight.fill(PLAYER1_COLOR)
                    elif (i, j) in self.player2_cells:
                        cell_highlight = pygame.Surface(
                            (CELL_SIZE - 2 * PADDING, CELL_SIZE - 2 * PADDING),
                            pygame.SRCALPHA,
                        )
                        cell_highlight.fill(PLAYER2_COLOR)
                
                # Blit the highlight if there is one
                if cell_highlight:
                    self.screen.blit(cell_highlight, (x + PADDING, y + PADDING))

        # Draw footer with score and pairs info
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            (0, HEADER_HEIGHT + self.grid.n * CELL_SIZE, self.width, FOOTER_HEIGHT),
        )

        footer_y = HEADER_HEIGHT + self.grid.n * CELL_SIZE
        
        # Calculate and display scores separately for players in multiplayer modes
        if self.game_mode == "vs_player" or self.game_mode == "vs_ai":
            # Calculate scores for each player
            player1_score = self.calculate_player_score(1)
            player2_score = self.calculate_player_score(2)
            
            # Player 1 score and pairs
            p1_score_text = self.font.render(
                f"{'Your' if self.game_mode == 'vs_ai' else 'Player 1'} Score: {player1_score}", 
                True, 
                (0, 128, 0)
            )
            p1_pairs_text = self.info_font.render(
                f"Pairs: {len(self.player1_pairs)}", 
                True, 
                (0, 128, 0)
            )
            
            # Player 2 / AI score and pairs
            p2_score_text = self.font.render(
                f"{'AI' if self.game_mode == 'vs_ai' else 'Player 2'} Score: {player2_score}", 
                True, 
                (208, 0, 0)
            )
            p2_pairs_text = self.info_font.render(
                f"Pairs: {len(self.player2_pairs)}", 
                True, 
                (208, 0, 0)
            )
            
            # Position and render player stats
            self.screen.blit(p1_score_text, (20, footer_y + 15))
            self.screen.blit(p1_pairs_text, (20, footer_y + 45))
            
            self.screen.blit(p2_score_text, (self.width - p2_score_text.get_width() - 20, footer_y + 15))
            self.screen.blit(p2_pairs_text, (self.width - p2_pairs_text.get_width() - 20, footer_y + 45))
            
        else:  # Solo mode
            # Calculate and display total score
            score = self.calculate_score()
            score_text = self.font.render(f"Score: {score}", True, TEXT_COLOR)
            pairs_text = self.info_font.render(
                f"Pairs: {len(self.pairs)}", True, TEXT_COLOR
            )
            
            # Position and render score
            self.screen.blit(score_text, (20, footer_y + 15))
            self.screen.blit(pairs_text, (20, footer_y + 45))
            
        # Display game status (if game is over)
        if self.game_over:
            status_text = self.font.render("Game Finished!", True, (50, 50, 200))
            self.screen.blit(status_text, (
                self.width // 2 - status_text.get_width() // 2,
                footer_y + 15
            ))
            
            result_text = self.info_font.render("Press SPACE to see results", True, TEXT_COLOR)
            self.screen.blit(result_text, (
                self.width // 2 - result_text.get_width() // 2,
                footer_y + 45
            ))
        
        # Display controls
        controls_text = self.info_font.render(
            "R: Reset  |  M: Menu  |  ESC: Quit", True, TEXT_COLOR
        )
        
        # Position at the bottom center
        self.screen.blit(
            controls_text,
            (
                self.width // 2 - controls_text.get_width() // 2,
                footer_y + FOOTER_HEIGHT - controls_text.get_height() - 10,
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
        if self.game_mode == "solo":
            # In solo mode, use the global cells_in_pairs
            if cell1 in self.cells_in_pairs or cell2 in self.cells_in_pairs:
                return False
        else:
            # In multiplayer modes, use player-specific cell sets
            player_cells = self.get_player_cells(self.current_player)
            if cell1 in player_cells or cell2 in player_cells:
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
        # Check if this is valid according to the game rules
        if self.game_mode == "solo":
            # In solo mode, check against global cells_in_pairs
            can_pair = self.can_pair_cells(cell1, cell2)
        else:  # vs_player or vs_ai
            # In multiplayer, check against player-specific cells
            player_cells = self.get_player_cells(self.current_player)
            can_pair = (cell1 not in player_cells and 
                       cell2 not in player_cells and 
                       self.are_adjacent(cell1, cell2) and 
                       not self.grid.is_pair_forbidden([cell1, cell2]))
                       
        if can_pair:
            # Add to main pairs list
            self.pairs.append([cell1, cell2])
            
            # Add to player-specific pairs list based on current player
            if self.current_player == 1:
                self.player1_pairs.append([cell1, cell2])
                # Add to player1's cells set
                self.player1_cells.add(cell1)
                self.player1_cells.add(cell2)
            else:
                self.player2_pairs.append([cell1, cell2])
                # Add to player2's cells set
                self.player2_cells.add(cell1)
                self.player2_cells.add(cell2)
                
            # For backward compatibility with solo mode
            if self.game_mode == "solo":
                self.cells_in_pairs.add(cell1)
                self.cells_in_pairs.add(cell2)
            
            # Switch players in multiplayer modes
            if self.game_mode in ["vs_player", "vs_ai"]:
                # Switch to the other player
                self.current_player = 2 if self.current_player == 1 else 1
                
            # Check if game is over or if the next player has valid moves
            self.check_game_over()

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
        
    def calculate_player_score(self, player_num):
        """
        Calculate the score for a specific player.
        
        Args:
            player_num: 1 for player 1, 2 for player 2 or AI
            
        Returns:
            The player's score (cost of their pairs + half of unpaired cells)
        """
        pairs = self.player1_pairs if player_num == 1 else self.player2_pairs
        
        # Calculate cost of the player's chosen pairs
        pair_cost = sum(self.grid.cost(pair) for pair in pairs)
        
        # In multiplayer, remaining cells are those not paired by either player
        # Calculate all paired cells as union of player1 and player2 cells
        all_paired_cells = self.player1_cells.union(self.player2_cells)
        
        # Each player gets half of the unpaired cost
        unpaired_cost = 0
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) not in all_paired_cells and self.grid.get_coordinate_color(
                    i, j
                ) != "k":
                    unpaired_cost += self.grid.get_coordinate_value(i, j)
        
        # Split unpaired cost between players
        unpaired_cost = unpaired_cost / 2
        
        return pair_cost + unpaired_cost

    def check_game_over(self):
        """Check if the game is over (no more valid pairs possible for any player)."""
        # Check if valid pairs exist for any player
        valid_pairs_exist = False
        
        # Check for valid pairs that aren't already taken
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                # In vs_player or vs_ai mode, we use separate cell tracking for each player
                if self.game_mode in ["vs_player", "vs_ai"]:
                    if self.current_player == 1 and (i, j) in self.get_player_cells(1):
                        continue
                    if self.current_player == 2 and (i, j) in self.get_player_cells(2):
                        continue
                else:  # Solo mode
                    if (i, j) in self.cells_in_pairs:
                        continue
                    
                # Check all adjacent cells
                adjacents = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                for adj_i, adj_j in adjacents:
                    if not (0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m):
                        continue
                    
                    # In vs_player or vs_ai mode, check against player-specific cells
                    if self.game_mode in ["vs_player", "vs_ai"]:
                        if self.current_player == 1 and (adj_i, adj_j) in self.get_player_cells(1):
                            continue
                        if self.current_player == 2 and (adj_i, adj_j) in self.get_player_cells(2):
                            continue
                    else:  # Solo mode
                        if (adj_i, adj_j) in self.cells_in_pairs:
                            continue
                        
                    # Check if pair is valid according to color rules
                    if not self.grid.is_pair_forbidden([(i, j), (adj_i, adj_j)]):
                        # Found a valid pair
                        valid_pairs_exist = True
                        
                        # In solo mode, any valid pair means the game continues
                        if self.game_mode == "solo":
                            return  # Game continues
                        
                        # In vs modes, we continue checking for all valid pairs
                        if self.game_mode in ["vs_player", "vs_ai"]:
                            # Check if current player can make a valid move
                            if self.current_player_has_valid_move():
                                return  # Current player can move, continue game
        
        # If we reach here, either no valid pairs exist at all, or the current player can't move
        if self.game_mode in ["vs_player", "vs_ai"]:
            if valid_pairs_exist:
                # Valid pairs exist but current player can't move - switch players and check again
                self.current_player = 2 if self.current_player == 1 else 1
                if self.current_player_has_valid_move():
                    return  # Other player can move, continue game
            
            # If we get here, neither player can move or no valid pairs exist at all
            self.game_over = True
        else:
            # In solo mode, if no valid pairs were found, game is over
            self.game_over = True
        
        # Create game result if game is over
        if self.game_over:
            if self.game_mode == "solo":
                self.game_result = GameResult(
                    self.calculate_score(), 0, len(self.pairs), 0, 
                    self.game_mode
                )
            else:
                player1_score = self.calculate_player_score(1)
                player2_score = self.calculate_player_score(2)
                self.game_result = GameResult(
                    player1_score, player2_score, 
                    len(self.player1_pairs), len(self.player2_pairs),
                    self.game_mode, self.ai_algorithm
                )
    
    def current_player_has_valid_move(self):
        """Check if the current player has any valid moves available."""
        # Get the current player's cells to avoid
        if self.game_mode in ["vs_player", "vs_ai"]:
            player_cells = self.get_player_cells(self.current_player)
        else:
            player_cells = self.cells_in_pairs
        
        # Check for valid pairs
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) in player_cells:
                    continue
                    
                # Check all adjacent cells
                adjacents = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                for adj_i, adj_j in adjacents:
                    if not (0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m):
                        continue
                    if (adj_i, adj_j) in player_cells:
                        continue
                        
                    # If we find any valid pair, the current player can move
                    if not self.grid.is_pair_forbidden([(i, j), (adj_i, adj_j)]):
                        return True
                        
        # No valid moves found
        return False
        
    def reset_game(self):
        """Reset the game to initial state."""
        self.pairs = []
        self.player1_pairs = []
        self.player2_pairs = []
        self.selected_cell = None
        self.hovering_cell = None
        self.cells_in_pairs = set()
        self.player1_cells = set()
        self.player2_cells = set()
        self.game_over = False
        self.current_player = 1  # Player 1 starts

    def ai_move(self):
        """Execute a single AI move based on the selected algorithm."""
        # Skip if not AI's turn or game is over
        if self.current_player != 2 or self.game_over:
            return
            
        # Create a copy of the grid for the solver
        grid_copy = Grid.grid_from_file(self.grid_file, read_values=True)
        grid_copy.cell_init()
        
        # Initialize appropriate solver
        if self.ai_algorithm == "greedy":
            solver = SolverGreedy(grid_copy)
        elif self.ai_algorithm == "fulkerson":
            solver = SolverFulkerson(grid_copy)
        elif self.ai_algorithm == "hungarian":
            solver = SolverHungarian(grid_copy)
        else:
            return  # Invalid algorithm
        
        # Get AI's cells to avoid (AI is player 2)
        ai_cells = self.player2_cells
        
        # Find all valid available pairs (adjacent cells not already paired by AI)
        valid_pairs = []
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) in ai_cells:
                    continue
                
                # Check adjacent cells
                adjacents = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                for adj_i, adj_j in adjacents:
                    # Skip if out of bounds or already paired by AI
                    if not (0 <= adj_i < self.grid.n and 0 <= adj_j < self.grid.m):
                        continue
                    if (adj_i, adj_j) in ai_cells:
                        continue
                    
                    # Check if valid according to color rules
                    pair = [(i, j), (adj_i, adj_j)]
                    if not self.grid.is_pair_forbidden(pair):
                        valid_pairs.append(pair)
        
        # If no valid pairs left, end AI turn and check if game is over
        if not valid_pairs:
            self.current_player = 1  # Switch back to player
            self.check_game_over()
            return
        
        # For simple greedy selection - choose cheapest pair
        if self.ai_algorithm == "greedy":
            # Sort by cost
            valid_pairs.sort(key=self.grid.cost)
            best_pair = valid_pairs[0]
            self.add_pair(best_pair[0], best_pair[1])
            return  # add_pair will switch players and check game_over
        
        # For more advanced algorithms, run solver and find a good match
        # from its solution that's still available
        solver.run()
        
        # Create a set of solver pairs for faster lookup
        solver_pair_set = set()
        for pair in solver.pairs:
            # Add both orderings of each pair to ensure matching
            p1, p2 = pair
            solver_pair_set.add((p1, p2))
            solver_pair_set.add((p2, p1))
        
        # Find pairs that are both valid in the game and recommended by the solver
        for pair in valid_pairs:
            # Try both orderings
            if tuple(pair) in solver_pair_set:
                # Found a pair that's in the solver's solution
                self.add_pair(pair[0], pair[1])
                return  # add_pair will switch players and check game_over
        
        # If no solver-recommended pairs are available, just pick the cheapest valid pair
        valid_pairs.sort(key=self.grid.cost)
        best_pair = valid_pairs[0]
        self.add_pair(best_pair[0], best_pair[1])  # add_pair will switch players and check game_over

    def show_results_screen(self):
        """Show the game results screen."""
        if not self.game_result:
            return
            
        self.game_result.draw(self.screen, self.font, self.info_font, self.title_font)
    
    def run(self):
        """Run the main game loop."""
        running = True
        clock = pygame.time.Clock()

        while running:
            # If in menu mode, show menu
            if self.show_menu:
                self.draw_menu()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEMOTION:
                        mouse_pos = event.pos
                        for button in self.menu_buttons:
                            button.update(mouse_pos)
                        self.grid_dropdown.update(mouse_pos)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Try dropdown first
                        if self.grid_dropdown.handle_event(event):
                            continue
                            
                        # Then try buttons
                        for button in self.menu_buttons:
                            if button.handle_event(event):
                                break  # Button was clicked
            
            # Results screen
            elif self.show_results:
                self.show_results_screen()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_m or event.key == pygame.K_SPACE:
                            # Return to menu
                            self.show_menu = True
                            self.show_results = False
                            self.menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

            # Game mode
            else:
                # Check if it's AI's turn in vs_ai mode
                if self.game_mode == "vs_ai" and self.current_player == 2 and not self.game_over:
                    # Slight delay for better visualization
                    pygame.time.delay(500)
                    self.ai_move()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_m:
                            # Return to menu
                            self.show_menu = True
                            self.menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
                        elif event.key == pygame.K_SPACE and self.game_over:
                            # Show results screen
                            self.show_results = True

                    elif event.type == pygame.MOUSEMOTION:
                        # Update hovering cell
                        self.hovering_cell = self.get_cell_at_pos(event.pos)

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Allow player cell clicks if it's human player's turn and game not over
                        # In vs_ai mode, only player 1 is human
                        # In vs_player mode, both players are human
                        # In solo mode, player 1 is the only player
                        is_human_turn = (
                            (self.game_mode == "solo") or
                            (self.game_mode == "vs_player") or
                            (self.game_mode == "vs_ai" and self.current_player == 1)
                        )
                        
                        if event.button == 1 and is_human_turn and not self.game_over:
                            clicked_cell = self.get_cell_at_pos(event.pos)

                            if clicked_cell:
                                # If no cell is selected, select this one
                                if self.selected_cell is None:
                                    # Check if the cell is available for the current player
                                    cell_available = True
                                    if self.game_mode == "solo" and clicked_cell in self.cells_in_pairs:
                                        cell_available = False
                                    elif self.game_mode in ["vs_player", "vs_ai"]:
                                        player_cells = self.get_player_cells(self.current_player)
                                        if clicked_cell in player_cells:
                                            cell_available = False
                                    
                                    # Only select if available for this player
                                    if cell_available:
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

                                    # If can't pair, select the new cell if it's available
                                    else:
                                        cell_available = True
                                        if self.game_mode == "solo" and clicked_cell in self.cells_in_pairs:
                                            cell_available = False
                                        elif self.game_mode in ["vs_player", "vs_ai"]:
                                            player_cells = self.get_player_cells(self.current_player)
                                            if clicked_cell in player_cells:
                                                cell_available = False
                                                
                                        if cell_available:
                                            self.selected_cell = clicked_cell
                
                # Draw the updated grid
                self.draw_grid()

            # Cap the frame rate
            clock.tick(60)

        pygame.quit()


def main():
    """Main function to start the game."""
    # Create and run the game
    game = GridGame()
    game.run()


if __name__ == "__main__":
    main()
