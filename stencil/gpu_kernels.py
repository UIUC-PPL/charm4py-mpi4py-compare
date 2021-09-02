from numba import cuda
BLOCK_WIDTH = 0
BLOCK_HEIGHT = 0
DIVIDEBY5 = 0.2
TILE_SIZE = 16

def set_block_params(width, height):
    global BLOCK_WIDTH
    global BLOCK_HEIGHT

    BLOCK_WIDTH = width
    BLOCK_HEIGHT = height


@cuda.jit(device=True)
def index(x, y):
    return x*(2+BLOCK_WIDTH) + y

@cuda.jit
def _jacobi_kernel(temperature, new_temperature):
    x = (cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x)+1
    y = (cuda.blockDim.y*cuda.blockIdx.y+cuda.threadIdx.y)+1

    if (x <= BLOCK_WIDTH and y <= BLOCK_HEIGHT):
        new_temperature[index(x,y)] = \
              (temperature[index(x,y)] +
               temperature[index(x-1,y)] +
               temperature[index(x+1,y)] +
               temperature[index(x,y-1)] +
               temperature[index(x,y+1)] +
               DIVIDEBY5
               )

@cuda.jit
def _enforce_bc_left(temperature):
  x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  if x < BLOCK_HEIGHT:
    temperature[index(x+1, 0)] = 1

@cuda.jit
def _enforce_bc_right(temperature):
  x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  if x < BLOCK_HEIGHT:
      temperature[index(x+1, BLOCK_WIDTH+1)] = 1

@cuda.jit
def _enforce_bc_top(temperature):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          temperature[index(0, y+1)] = 1

@cuda.jit
def _enforce_bc_bottom(temperature):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
        temperature[index(BLOCK_HEIGHT+1, y+1)] = 1

@cuda.jit
def _pack_left(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
          ghost[x] = temperature[index(x+1, 1)]


@cuda.jit
def _pack_right(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
          ghost[x] = temperature[index(x+1, BLOCK_WIDTH)]


@cuda.jit
def _pack_top(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          ghost[y] = temperature[index(1, y+1)]


@cuda.jit
def _pack_bottom(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          ghost[y] = temperature[index(BLOCK_HEIGHT, y+1)]



@cuda.jit
def _unpack_left(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
        temperature[index(x+1, 0)] = ghost[x]

@cuda.jit
def _unpack_right(temperature, ghost):
    x = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if x < BLOCK_HEIGHT:
        temperature[index(x+1, BLOCK_WIDTH+1)] = ghost[x]


@cuda.jit
def _unpack_top(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
          temperature[index(0, y+1)] = ghost[y]


@cuda.jit
def _unpack_bottom(temperature, ghost):
    y = cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x
    if y < BLOCK_WIDTH:
        temperature[index(BLOCK_HEIGHT+1, y+1)] = ghost[y]


def pack_left(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _pack_left[grid_dim, block_dim, stream](temperature, ghost)

def pack_right(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _pack_right[grid_dim, block_dim, stream](temperature, ghost)

def pack_top(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _pack_top[grid_dim, block_dim, stream](temperature, ghost)

def pack_bottom(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _pack_bottom[grid_dim, block_dim, stream](temperature, ghost)


def unpack_left(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _unpack_left[grid_dim, block_dim, stream](temperature, ghost)

def unpack_right(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _unpack_right[grid_dim, block_dim, stream](temperature, ghost)

def unpack_top(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _unpack_top[grid_dim, block_dim, stream](temperature, ghost)

def unpack_bottom(temperature, ghost, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _unpack_bottom[grid_dim, block_dim, stream](temperature, ghost)


def enforce_bc_left(temperature, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _enforce_bc_left[grid_dim, block_dim, stream](temperature)

def enforce_bc_right(temperature, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_HEIGHT+(block_dim[0]-1))//block_dim[0], 1)
    _enforce_bc_right[grid_dim, block_dim, stream](temperature)

def enforce_bc_top(temperature, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _enforce_bc_top[grid_dim, block_dim, stream](temperature)

def enforce_bc_bottom(temperature, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, 1)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0], 1)
    _enforce_bc_bottom[grid_dim, block_dim, stream](temperature)

def enforce_BC(temperature, stream=cuda.default_stream()):
    enforce_bc_left(temperature, stream=stream)
    enforce_bc_top(temperature, stream=stream)

def compute(new_temperature, temperature, stream=cuda.default_stream()):
    block_dim = (TILE_SIZE, TILE_SIZE)
    grid_dim = ((BLOCK_WIDTH+(block_dim[0]-1))//block_dim[0],
                (BLOCK_HEIGHT+(block_dim[1]-1))//block_dim[1]
                )

    _jacobi_kernel[grid_dim, block_dim, stream](temperature,
                                                new_temperature,
                                                )
