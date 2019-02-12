from pandas import DataFrame
import numpy as np
cimport numpy as np

cdef enum states:
    clean = 0
    dirty = 1

murphys_law = False

cdef np.ndarray gen_world(n_init_dirty):
    world = np.arange(9, dtype=np.intc)
    np.random.shuffle(world)

    for i in range(n_init_dirty):
        world[np.argmax(world)] = -1

    for i in range(9):
        if world[i] < 0:
            world[i] = dirty
        else:
            world[i] = clean

    return world.reshape(3,3)

cdef class Agent:
    cdef dict movement_table
    cdef (int, int) loc

    def __cinit__(Agent self):
        self.loc = (np.random.randint(3),np.random.randint(3))
        self.movement_table = {
                (0,0):(0,1), (0,1):(0,2), (0,2):(1,2),
                (1,0):(0,0), (1,1):(1,2), (1,2):(2,2),
                (2,0):(1,0), (2,1):(2,0), (2,2):(2,1)
                }

    def act(Agent self, np.ndarray world):
        dirt_sensor = world[self.loc] == dirty
        if murphys_law and not np.random.randint(0,10):
            dirt_sensor = not dirt_sensor
        if dirt_sensor:
            world[self.loc] = clean
            if murphys_law and not np.random.randint(0,4):
                world[self.loc] = dirty
        else:
            self.loc = self.movement_table[self.loc]

cdef class RandomAgent:
    cdef (int, int) loc

    def __cinit__(RandomAgent self):
        self.loc = (np.random.randint(3),np.random.randint(3))

    def act(RandomAgent self, np.ndarray world):
        rint = lambda a,b: np.random.random_integers(a,b)
        if rint(0,1): # clean
            world[self.loc] = clean
            if murphys_law and not rint(0,3):
                world[self.loc] = dirty
        elif rint(0,1): # move right/left
            self.loc = self.loc[0], self.loc[1] + rint(-1,1)
        else: # move up/down
            self.loc = self.loc[0] + rint(-1,1), self.loc[1]

        self.validate_loc()

    cdef void validate_loc(RandomAgent self):
        if self.loc[0] < 0:
            self.loc[0] = 0
        elif self.loc[0] > 2:
            self.loc[0] = 2
        if self.loc[1] < 0:
            self.loc[1] = 0
        elif self.loc[1] > 2:
            self.loc[1] = 2

cdef double trial(agent_constructor, n_init_dirty,
        int n_trials=10000, int n_turns=20):
    cdef int cntr
    cdef np.ndarray world

    cntr = 0
    for i in range(n_trials):
        world = gen_world(n_init_dirty)
        agent = agent_constructor()
        for j in range(n_turns):
            agent.act(world)
        cntr += 9 - np.sum(world)

    return cntr / <float>n_trials


def main():
    cdef np.ndarray results = np.empty((4,3))
    cdef Py_ssize_t i
    for i in range(3):
        results[0][i] = trial(Agent, i*2+1)
        results[1][i] = trial(RandomAgent, i*2+1)

    global murphys_law
    murphys_law = True

    for i in range(3):
        results[2][i] = trial(Agent, i*2+1)
        results[3][i] = trial(RandomAgent, i*2+1)

    results = np.around(results, decimals=2)
    print(DataFrame(results,["i","ii","iii","iv"],[1,3,5]))
