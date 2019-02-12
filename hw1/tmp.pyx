%%cython

import numpy as np
cimport numpy as np

world_size = 0
n_init_dirty = 3
world_shape = (3,3)

def gen_world():
    global world_size
    world_size = world_shape[0] * world_shape[1]

    world = np.arange(world_size)
    np.random.shuffle(world)

    for i in range(n_init_dirty):
        world[np.argmax(world)] = -1

    for i in range(world_size):
        if world[i] < 0:
            world[i] = 1
        else:
            world[i] = 0

    return world.reshape(world_shape)

def move(current, change):
    for i in range(len(current)):
        current[i] += change[i]

world = gen_world()

rule_table = {}
for i in range(world_shape[0]):
    for j in range(world_shape[1]):
        rule_table[i,j,clean] =
        rule_table[i,j,dirty] = "suck"

precept = (None,None)
location = np.random.randint(world_shape[0],size=2)


