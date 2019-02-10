# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

import numpy as np
cimport numpy as np
import cython
import sys
from collections import defaultdict

from libc.math cimport abs as c_abs
from libc.stdio cimport printf

DEF DEBUG = 2


ctypedef int (*heuristic_func)(int[:,::1]) # cython function pointer
ctypedef unsigned long long int ulint # for ids

cdef int misplaced_tiles_heuristic(int[:,::1] board) nogil:
    cdef:
        size_t i, j
        int idim
        int count = 0
        int dim = board.shape[0] # should be same as shape[1]

    for i in range(dim):
        idim = i * dim
        for j in range(dim):
            if board[i,j] == 0: continue
            if board[i,j]-1 != idim + j:
            # board[i,j]-1: -1 corrects for indexing (1 belongs at index 0)
                count += 1

    return count

@cython.cdivision(True)
cdef int distance_tiles_heuristic(int[:,::1] board):# nogil:
    cdef:
        long i, j
        int val
        int count = 0
        int dim = board.shape[0] # should be same as shape[1]

    for i in range(dim):
        for j in range(dim):
            val = board[i,j]
            if val == 0: continue
            val -= 1 # correct for indexing
            count += c_abs(val / dim - i) # distance in rows
            count += c_abs(val % dim - j) # distance in columns

    return count

cdef int num_inversions_heuristic(int[:,::1] tmp):
# count the number of tiles that are before a smaller tile
    cdef:
        size_t i, j, k = 0
        int count = 0
        int[::1] board = np.empty(tmp.shape[0] * tmp.shape[1],dtype=np.intc)

    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            board[k] = tmp[i,j]
            k += 1

    for i in board:
        for j in board[i:]:
            if j == 0: continue
            if j < i:
                count += 1
    return count

cdef heuristic_func heuristic_table[3]
heuristic_table[0] = misplaced_tiles_heuristic
heuristic_table[1] = distance_tiles_heuristic
heuristic_table[2] = num_inversions_heuristic


cdef ulint board_config_to_id(int[:,::1] board):
    cdef int[::1] row
    cdef int i
    cdef ulint place = 1
    cdef ulint _hash = 0

    for row in board:
        for i in row:
            _hash += i * place
            place *= 11

    return _hash


cdef class Node:
    cdef:
        int[:,::1] board
        int hscore
        int fscore
        int gscore
        Node parent
        ulint id

    def __cinit__(Node self,
            int[:,::1] current_state,
            size_t heuristic_num,
            ulint id,
            int gscore = 0,
            Node parent = None
            ):
        self.board = np.copy(current_state)

        self.hscore = heuristic_table[heuristic_num](current_state)
        self.gscore = gscore
        self.fscore = gscore + self.hscore

        self.id = board_config_to_id(current_state)

        if parent:
            self.parent = parent
        else:
            self.parent = self

    def __lt__(Node self, Node other): return self.fscore <  other.fscore
    def __gt__(Node self, Node other): return self.fscore >  other.fscore
    def __le__(Node self, Node other): return self.fscore <= other.fscore
    def __ge__(Node self, Node other): return self.fscore >= other.fscore
    def __eq__(Node self, Node other): return self.fscore == other.fscore
    def __ne__(Node self, Node other): return self.fscore != other.fscore


    cdef list backtrace(Node self):
    # return path from node to most distant parent (starting node)
        if self.parent.id == self.id:
            return [self.board]
        return self.parent.backtrace() + [self.board]

cdef (size_t,size_t) get_empty_tile_pos(int[:,::1] board):
    cdef size_t i, j

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i,j] == 0:
                return (i,j)

    raise Exception("No empty tile in board")

cdef enum direction_enum:
# for readability
    up = 0
    down = 1
    left = 2
    right = 3

cdef void slide_tile(int[:,::1] board, direction_enum direction):
# direction is the movement direction of the tile being
# moved into empy space, ex: 1 0 -> 1 2 :this board config has
#                            3 2 -> 3 0 :had an "up" slide
    cdef size_t i, j
    i,j = get_empty_tile_pos(board)


    if direction == up:
        board[i,j] = board[i+1,j]
        board[i+1,j] = 0
    elif direction == down:
        board[i,j] = board[i-1,j]
        board[i-1,j] = 0
    elif direction == left:
        board[i,j] = board[i,j-1]
        board[i,j-1] = 0
    else: # right
        board[i,j] = board[i,j+1]
        board[i,j+1] = 0

cdef void unslide_tile(int[:,::1] board, direction_enum direction):
    cdef size_t i, j
    i,j = get_empty_tile_pos(board)

    if direction == up:
        board[i,j] = board[i-1,j]
        board[i-1,j] = 0
    elif direction == down:
        board[i,j] = board[i+1,j]
        board[i+1,j] = 0
    elif direction == left:
        board[i,j] = board[i,j+1]
        board[i,j+1] = 0
    else: # right
        board[i,j] = board[i,j-1]
        board[i,j-1] = 0

cdef bint is_valid_move(int[:,::1] board, direction_enum direction):
    cdef size_t i, j
    i,j = get_empty_tile_pos(board)

    if direction == up:     return i < board.shape[0] - 1
    elif direction == down: return i > 0
    elif direction == left: return j > 0
    else:                   return j < board.shape[1] - 1 # else right

cdef int[:,::1] create_goal_state(int[:,::1] board):
    cdef np.ndarray tmp

    # create winning board config
    tmp = np.arange(board.shape[0]*board.shape[1], dtype=np.intc)
    tmp += 1
    tmp[tmp.shape[0] - 1] = 0
    tmp = tmp.reshape(board.shape[0], board.shape[1])

    return tmp

cdef ulint create_goal_state_id(int[:,::1] board):
    return board_config_to_id(create_goal_state(board))

cpdef A_star(int[:,::1] board, size_t heuristic_num=1):
# returns list of path nodes on best path through state graph
    cdef:
        direction_enum direction
        int[:,::1] adj_state # adjacent state
        ulint adj_state_id
        ulint goal_id
        Node current
        Node new_node
        Node start
        int g
        dict open_set
        dict closed_set = dict()
        unsigned long loop_counter = 5_000_000 # just in case

    assert is_solvable(np.asarray(board)), \
            "Config Error: goal state not reachable from initial state"

    start = Node(board, heuristic_num, id=board_config_to_id(board))
    goal_id = create_goal_state_id(board)

    open_set = {start.id : start}

    adj_state = np.empty((board.shape[0], board.shape[1]), dtype=np.intc)

    IF DEBUG >= 2: print("goal_id:",goal_id)

    # "graph" traversal loop
    #while open_set and loop_counter:
    while open_set:
        current = min(open_set.values())

        IF DEBUG == 1: printf("%ld; ",loop_counter)
        IF DEBUG >= 2:
            printf("%ld;",loop_counter)
            printf("\th: %d;",current.hscore)
            printf("\tf: %d;",current.fscore)
            #printf("\t|open_set|: %ld;",len(open_set))
            #print("current.id:",current.id, end=";  ")
            #print(".parent.id: ",current.parent.id, end=";  ")
            printf('\n')

        if current.id == goal_id: return current.backtrace()

        del open_set[current.id]
        closed_set[current.id] = current

        adj_state[:,:] = current.board[:,:]

        # only four possible directions/adjacent states for any state
        for direction in range(4):
            if not is_valid_move(adj_state, direction): continue
            slide_tile(adj_state, direction)
            adj_state_id = board_config_to_id(adj_state)

            # move over states which have already been finalized
            if adj_state_id in closed_set:# or adj_state_id == current.parent.id:
                unslide_tile(adj_state, direction)
                continue

            # distance from start to new_board_state
            g = current.gscore + 1

            if adj_state_id not in open_set:
                # discoverd new state
                new_node = Node(adj_state,
                                heuristic_num,
                                gscore=g,
                                parent=current,
                                id=adj_state_id)
                open_set[adj_state_id] = new_node
            else:
                new_node = open_set[adj_state_id]
                if g < new_node.gscore:
                    new_node.parent = current
                    new_node.gscore = g
                    new_node.fscore = g + new_node.hscore

            unslide_tile(adj_state, direction)

        loop_counter -= 1

    if loop_counter == 0:
        raise Exception("Error in A_star: loop count exceeded")
    raise Exception("Exited loop without finding goal state")

cpdef list best_first_search(int[:,::1] board, size_t heuristic_num=1):
    cdef:
        direction_enum direction
        int[:,::1] goal
        int[:,::1] adj_state # adjacent state
        int[:,::1] next_best
        unsigned int max_h
        unsigned int tmp_h
        list path
        unsigned long loop_counter = 5_000_000 # just in case
        heuristic_func heuristic = heuristic_table[heuristic_num]

    assert is_solvable(np.asarray(board)), \
            "Config Error: goal state not reachable from initial state"

    goal = create_goal_state(board)
    adj_state = np.empty((board.shape[0], board.shape[1]), dtype=np.intc)
    next_best = np.empty((board.shape[0], board.shape[1]), dtype=np.intc)
    adj_state[:,:] = board[:,:]
    path = [adj_state.copy()]

    # "graph" traversal loop
    while loop_counter:
        IF DEBUG >= 2: printf("%ld\t",loop_counter)

        if np.array_equal(adj_state, goal): return path

        # only four possible directions/adjacent states for any state
        min_h = -1
        for direction in range(4):
            if not is_valid_move(adj_state, direction): continue

            slide_tile(adj_state, direction)

            if np.array_equal(adj_state, path[len(path)-1]):
                unslide_tile(adj_state, direction)
                continue

            tmp_h = heuristic(adj_state)

            if tmp_h < min_h:
                next_best[:,:] = adj_state[:,:]
                min_h = tmp_h

            unslide_tile(adj_state, direction)

        adj_state[:,:] = next_best[:,:]
        path.append(adj_state.copy())

        loop_counter -= 1

    if loop_counter == 0:
        raise Exception("Error in A_star: loop count exceeded")
    raise Exception("Exited loop without finding goal state")

cpdef void display_path(list lpath):
    cdef int[:,::1] board
    print("Solved in",len(lpath)-1,"moves:")
    for i,board in enumerate(lpath):
        print("State:",i+1)
        print(np.asarray(board))

@cython.cdivision(True)
cpdef bint is_solvable(np.ndarray board):
    cdef size_t i,j, empty_tile_row = -1
    cdef int[::1] checker = board.flatten()
    cdef int inversions = 0
    cdef Py_ssize_t n = board.shape[0] # board is nxn dimensions

    for i in range(checker.shape[0]):
        for j in range(i, checker.shape[0]):

            if checker[j] == 0:      # found empty slot
                empty_tile_row = j / n # record location(row)
                continue               # no comparisons

            if checker[i] > checker[j]:
                inversions += 1

    if n % 2:                     # if n, in nxn board, is odd
        return not inversions % 2       # then inversions should be odd

    if empty_tile_row % 2:        # if empty slot in odd indexed row
        return not inversions % 2   # then inversions should be even
    else:                         # if empty slot in even indexed row
        return inversions % 2       # then inversions should be odd


cdef void run_test(int puzzle_num, np.ndarray board, size_t h_num):
    print("\nTest #",puzzle_num,"heuristic",h_num)
    print("initial state:")
    print(board)
    display_path(A_star(board,heuristic_num=1))

cdef int run_tests_2x2(int t_num, size_t h_num): # contains 8 tests
    cdef np.ndarray board

    t_num += 1
    board = np.asarray([[1,0],
                        [3,2]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[0,1],
                        [3,2]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[3,1],
                        [0,2]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[3,1],
                        [2,0]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[3,0],
                        [2,1]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[0,3],
                        [2,1]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[2,3],
                        [0,1]], dtype=np.intc)
    run_test(t_num, board, h_num)

    t_num += 1
    board = np.asarray([[2,3],
                        [1,0]], dtype=np.intc)
    run_test(t_num, board, h_num)

    return t_num

def run_tests():
    cdef np.ndarray board
    cdef int t_num = 0

    t_num = run_tests_2x2(t_num,2) # contains 8 tests

    t_num += 1
    board = np.asarray([[1,2,3],
                        [4,5,0],
                        [7,8,6]], dtype=np.intc)
    run_test(t_num, board, 2)

    ### impossible config
#    t_num += 1
#    board = np.asarray([[4,5,0],[6,1,8],[7,3,2]], dtype=np.intc)
#    run_test(t_num, board, 2)

    t_num += 1
    board = np.asarray([ [ 6, 5, 2,15],
                         [12, 9,14, 4],
                         [ 1,10,11,13],
                         [ 3, 8, 0, 7]], dtype=np.intc)
    run_test(t_num, board, 2)


    IF DEBUG: print("END")



























from math import sqrt
def get_board_dim(board_size): # must be square
    dim = sqrt(board_size)
    if dim % 1 != 0:
        sys.exit("Board size must be square:\n" + \
                f"board_size: {board_size}\n")
    return ( np.int_(dim), np.int_(dim) )

@cython.wraparound(True)
def randomize_board(board_size):
    dim = get_board_dim(board_size)

    board = np.arange(board_size, dtype=np.intc)
    np.random.shuffle(board)

    board = board.reshape(dim)

    if not is_solvable(board):
        # bad config can be fixed with +/- 1 inversion
        if board[-1][-1] == 0:
            board[-1][-3], board[-1][-2] =  board[-1][-2], board[-1][-3]
        elif board[-1][-2] == 0:
            board[-1][-1], board[-1][-3] =  board[-1][-3], board[-1][-1]
        else:
            board[-1][-1], board[-1][-2] =  board[-1][-2], board[-1][-1]

    return board

cdef extern from *:
    ctypedef int int128 "__int128_t"

cdef ulint hash(int[:,::1] board):
#cdef int128 hash(int[:,::1] board):
    cdef int[::1] row
    cdef int i
    cdef ulint place = 1
    cdef ulint _hash = 0
    #cdef int128 place = 1
    #cdef int128 _hash = 0

    for row in board:
        for i in row:
            _hash += i * place
            place *= 100

    return _hash

def hash_test():
    cdef dict d = dict()
    #cdef int128 _hash
    cdef ulint _hash

    print(sizeof(ulint))
    print(sizeof(np.uint))
    print(sizeof(int128))

    board = np.asarray([ [ 6, 5, 2,15],
                         [ 1,10,14, 4],
                         [12, 9,11,13],
                         [ 3, 8, 0, 7]], dtype=np.intc)
    print(hash(board))
    board = np.asarray([ [ 6, 5, 2,15],
                         [12, 9,14, 4],
                         [ 1,10,11,13],
                         [ 3, 8, 0, 7]], dtype=np.intc)
    print(hash(board))

    board = create_goal_state(board)
    board[3][3] = 16 # checking for overflow
    print(hash(board))

    while True:
        print(len(d))
        board = ''
        _hash = 0
        #_hash = (0,0)
        if d: del d[0]
        else: d = dict()

        while _hash not in d:
            d[_hash] = board

            board = randomize_board(16)
            _hash = hash(board)


        if not np.array_equal(board,d[_hash]):
            print("\nfail")
            print(_hash)
            print(board)
            print(d[_hash])
            print()
            return

