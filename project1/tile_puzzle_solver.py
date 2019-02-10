import sys
import numpy as np
from math import sqrt
# import all python-accessible objects from cython module
from tile_puzzle_solver import *


init_config_error_msg = "enter initial puzzle configuration " + \
            "in format \"b n_1 n_2 . . . n_i-1\" (with quotes) " + \
            "where 'b' is the blank cell, i is a square number " + \
            "(or '-r <board_size>' for random config)"


def make_board(argv):
    # check init style
    init_config = argv[1]
    if init_config == "-r" or init_config == 'r':
        return randomize_board(int(argv[2]))

    # remove whitespace
    init_config = init_config.strip()

    # remove parens
    if init_config[0] == '(' and init_config[-1] == ')':
        init_config = init_config[1:-1].strip()

    # list of numbers (still of type str)
    init_config = init_config.split(' ')

    board_size = len(init_config) # must be square

    init_config = [int(i) if i != 'b' else 0 for i in init_config]

    board = np.asarray(init_config,dtype=np.intc)

    return board.reshape(get_board_dim(board_size))#, board_size

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit(init_config_error_msg)

    #board = make_board(sys.argv)

    #solution_path = A_star(board, 0)
    #display_path(solution_path)

    #solution_path = A_star(board, 1)
    #display_path(solution_path)

    #solution_path = A_star(board, 2)
    #display_path(solution_path)

    #solution_path = best_first_search(board, 2)
    #display_path(solution_path)
