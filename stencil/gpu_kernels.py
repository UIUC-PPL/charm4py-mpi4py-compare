from numba import cuda
BLOCK_WIDTH = 0
BLOCK_HEIGHT = 0
DIVIDEBY5 = 0.2

@cuda.jit
def index(x, y):
    return x*BLOCK_WIDTH + y


@cuda.jit
def pack_left(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
          ghost[x] = temperature[index(x+1, 0)]

@cuda.jit
def pack_right(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
          ghost[x] = temperature[index(x+1, BLOCK_WIDTH-1)]


@cuda.jit
def pack_top(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          ghost[y] = temperature[index(0, y+1)]


@cuda.jit
def pack_bottom(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          ghost[y] = temperature[index(BLOCK_HEIGHT-1, y+1)]


# @cuda.jit
# def unpack_left(temperature, ghost):
#     for x in range(BLOCK_HEIGHT):
#         temperature[index(x+1, 0)] = ghost[x]

# @cuda.jit
# def unpack_right(temperature, ghost):
#     for x in range(BLOCK_HEIGHT):
#         temperature[index(x+1,BLOCK_WIDTH-1)] = ghost[x]

# @cuda.jit
# def unpack_top(temperature, ghost):
#     for y in range(BLOCK_WIDTH):
#         temperature[index(0, y+1)] = ghost[y]

# @cuda.jit
# def unpack_bottom(temperature, ghost):
#     for y in range(BLOCK_WIDTH):
#         temperature[index(BLOCK_HEIGHT-1,y+1)] = ghost[y]
