import sys
import numpy as np

# import all python-accessible objects from cython module
from tile_puzzle_solver import *


heuristic_num = 1 # change to test different heuristics: 0, 1, or 2
                  # 0 for misplaced tiles heuristic
                  # 1 for manhattan distance heuristic
                  # 2 for num inversions heruistic

search_type = 1   # change to test diff
                  # 1 for 'A* search'
                  # 0 for 'Best-First search'

if search_type == 1:
    search = A_star
    search_name = "A* search"
elif search_type == 0:
    search = best_first_search
    search_name = "Best-First search"

if heuristic_num == 0:
    h_name = "Misplaced Tiles Heuristic"
elif heuristic_num == 1:
    h_name = "Manhattan Distance Heuristic"
elif heuristic_num == 2:
    h_name = "Num Inversions Heuristic"


init_config_error_msg = "enter initial puzzle configuration " + \
            "in format \"b n_1 n_2 . . . n_i-1\" (with quotes) " + \
            "where 'b' is the blank cell, i is a square number " + \
            "(or '-r <board_size>' for random config)"


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit(init_config_error_msg)

    if sys.argv[1] == "--runtests":
        print("running tests")
        run_tests()

    else:
        board = make_board(sys.argv)

        print("Solving puzzle with", search_name)
        print("\tusing",h_name)
        solution = search(board, heuristic_num)
        display_path(solution)
