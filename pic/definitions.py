from numba.experimental import jitclass
import numba
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError

PRK_M_PI = 3.14159265358979323846264338327950288419716939937510
MASS_INV = 1.0
Q = 1.0
epsilon = 0.000001
DT = 1.0
MEMORYSLACK = 10

REL_X = 0.5
REL_Y = 0.5

GEOMETRIC = 10
SINUSOIDAL = 11
LINEAR = 12
PATCH = 13
UNDEFINED = 14

PARTICLE_FIELDS = 10
PARTICLE_X = 0
PARTICLE_Y = 1
PARTICLE_VX = 2
PARTICLE_VY = 3
PARTICLE_Q = 4
PARTICLE_X0 = 5
PARTICLE_Y0 = 6
PARTICLE_K = 7
PARTICLE_M = 8
PARTICLE_ID = 9

@jitclass
class BoundingBox:
    left: int
    right: int
    bottom: int
    top: int

    def __init__(self, left: int, right: int, bottom: int, top: int):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

def enforce(x, err_msg):
    if not x:
        raise ArgumentTypeError(err_msg)

def parsed_iters(iters):
    iters = int(iters)
    enforce(iters > 0,
            f"Number of time steps must be positive: {iters}"
            )
    return iters

def parsed_grid_size(grid_size):
    L = int(grid_size)
    enforce(L > 0 and L % 2 == 0,
            f"Grid size (L) must be positive and even: {L}"
            )
    return L

def parsed_n(n):
    n = int(n)
    enforce(n > 0,
            f"Number of particles must be positive: {n}"
            )
    return n


def parsed_period(i):
    p = int(i)
    enforce(p > 0,
            f"Period must be positive: {p}"
            )
    return p

def parsed_k(i):
    return int(i)

def parsed_d(i):
    d = int(i)
    enforce(d > 0,
            f"Migration delay must be positive: {d}"
            )
    return d

def parsed_mode(i):
    enforce(i.lower() == 'geometric',
            "Currently only geometric mode is implemented"
            )
    return i

def parsed_yvelocity(velocity):
    return int(velocity)

def parse_args(argv):
    argp = ArgumentParser(description='PIC PRK For Python')
    argp.add_argument('-i', '--iterations',
                      help="The number of iterations in the simulation.",
                      type=parsed_iters
                      )
    argp.add_argument('-s', '--grid_size',
                      help="The number of cells in the grid.",
                      type=parsed_grid_size
                      )
    argp.add_argument('-n', '--num_particles',
                      help="The number of particles in the simulation.",
                      type=parsed_n
                      )
    argp.add_argument('-p', '--period',
                      help="The period (???) of the simulation.",
                      type=parsed_period
                      )
    argp.add_argument('-k', '--charge_increment',
                      help="Particle charge semi-increment",
                      type=parsed_k
                      )
    argp.add_argument('-m', '--velocity_y',
                      help="Initial vertical particle velocity.",
                      type=parsed_yvelocity
                      )
    argp.add_argument('-d', '--migration_delay',
                      help="The number of timesteps between migration attempts",
                      type=parsed_d
                      )
    argp.add_argument('--mode',
                      help="The mode of initialization, can be one of the "
                      "following support modes: GEOMETRIC",
                      type=parsed_mode
                      )
    argp.add_argument('--init_parameters',
                      help="Additional parameters for the initialization mode. "
                      "The currently-supported modes and their parameters are: "
                      "GEOMETRIC: attenuation factor.\n",
                      action='append'
                      )
    argp.add_argument("-v", "--verbose",
                      help="Output additional information for the run.",
                      action='store_true'
                      )
    return argp.parse_args()

def validate_args(parsed_args: dict):
    enforce(parsed_args.migration_delay < parsed_args.period,
            "Migration delay must be less than period, but: "
            f"{parsed_args.migration_delay} > {parsed_args.period}"
            )
