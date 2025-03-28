from hungarian import SolverHungarianChibre as Hungarian
from grid import Grid


def main():
    # Load grid from file
    data_path = "./input/"
    file_name = data_path + "grid21.in"
    grid = Grid.grid_from_file(file_name, read_values=True)

    # Initialize the Hungarian solver
    solver = Hungarian(grid)
    # cost_matrix = solver.cost_matrix_init()

    # Run the Hungarian algorithm
    matching_pairs = solver.run()
    print("brick")
    score = solver.score()
    # Print the results
    print(grid, solver, matching_pairs, score)
    print(score)
    print("brk")


if __name__ == "__main__":
    main()
#     # main()
