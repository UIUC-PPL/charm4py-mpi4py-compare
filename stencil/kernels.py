from numba import jit, njit
BLOCK_WIDTH = 0
BLOCK_HEIGHT = 0
DIVIDEBY5 = 0.2

def set_block_params(width, height):
    global BLOCK_WIDTH
    global BLOCK_HEIGHT

    BLOCK_WIDTH = width
    BLOCK_HEIGHT = height

@njit
def index(x, y):
    return x*BLOCK_WIDTH + y

@njit
def pack_left(temperature, ghost):
        for x in range(BLOCK_HEIGHT):
            ghost[x] = temperature[index(x+1, 0)]

@njit
def pack_right(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        ghost[x] = temperature[index(x+1,BLOCK_WIDTH-1)]

@njit
def pack_top(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        ghost[y] = temperature[index(0, y+1)]

@njit
def pack_bottom(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        ghost[y] = temperature[index(BLOCK_HEIGHT-1, y+1)]

@njit
def unpack_left(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        temperature[index(x+1, 0)] = ghost[x]

@njit
def unpack_right(temperature, ghost):
    for x in range(BLOCK_HEIGHT):
        temperature[index(x+1,BLOCK_WIDTH-1)] = ghost[x]

@njit
def unpack_top(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        temperature[index(0, y+1)] = ghost[y]

@njit
def unpack_bottom(temperature, ghost):
    for y in range(BLOCK_WIDTH):
        temperature[index(BLOCK_HEIGHT-1,y+1)] = ghost[y]

@njit
def compute(new_temperature, temperature):
    for i in range(1, BLOCK_HEIGHT+1):
        for j in range(1, BLOCK_WIDTH+1):
            new_temperature[index(i, j)] = (temperature[index(i-1, j)] \
                                            +  temperature[index(i+1, j)] \
                                            +  temperature[index(i, j-1)] \
                                            +  temperature[index(i, j+1)] \
                                            +  temperature[index(i, j)]) \
                                            *  DIVIDEBY5

@njit
def enforce_BC(temperature):
    # heat the left and top faces of the block
    for y in range(1,BLOCK_WIDTH+1):
        temperature[index(0,y)] = 255.0
    for x in range(1,BLOCK_HEIGHT+1):
        temperature[index(x,0)] = 255.0
