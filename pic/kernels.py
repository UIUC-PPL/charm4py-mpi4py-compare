from definitions import *
import math
import random
from random_draw import *
from array import array
import numba
PY_ONLY = False

def njit(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

if not PY_ONLY:
    njit = numba.njit

RNG = LCG()

def initialize_grid(tile: BoundingBox):
    n_columns = tile.right-tile.left+1
    n_rows = tile.top-tile.bottom+1
    grid = np.ndarray((n_columns, n_rows), dtype=np.float64)

    for y in range(tile.bottom, tile.top+1):
        for x in range(tile.left, tile.right+1):
            grid[x-tile.left, y-tile.bottom] = Q if x % 2 == 0 else -Q
    return grid

def finish_particle_initialization(particles, num_particles_prefix):
    my_num_particles = len(particles)
    ID = num_particles_prefix - my_num_particles + 1

    for pi in range(my_num_particles):
        p = particles[pi]
        x_coord = p[PARTICLE_X]
        y_coord = p[PARTICLE_Y]
        rel_x = math.fmod(x_coord, 1.0)
        rel_y = math.fmod(y_coord, 1.0)
        r1_sq = rel_y * rel_y + rel_x * rel_x
        r2_sq = rel_y * rel_y + (1.0-rel_x) * (1.0-rel_x)
        cos_theta = rel_x/math.sqrt(r1_sq)
        cos_phi = (1.0-rel_x)/math.sqrt(r2_sq)
        base_charge = 1.0 / ((DT*DT) * Q * (cos_theta/r1_sq + cos_phi/r2_sq))

        p[PARTICLE_VX] = 0.0
        p[PARTICLE_VY] = p[PARTICLE_M] / DT
        PK = p[PARTICLE_K]
        p[PARTICLE_Q] = (2*PK+1)*base_charge

        x = int(x_coord)

        if x % 2:
            p[PARTICLE_Q] *= -1

        p[PARTICLE_X0] = x_coord
        p[PARTICLE_Y0] = y_coord
        p[PARTICLE_ID] = ID
        ID += 1



def initialize_geometric(n_input: int, L: int, rho: float, tile: BoundingBox,
                         k: float, m: float):

    n_placed = 0
    A = n_input * ((1.0-rho) / (1.0-math.pow(rho, L))) / L

    for x in range(tile.left, tile.right):
        start_index = tile.bottom+x*L
        RNG.jump(2*start_index, 0)
        # cleanup: this can be done in constant time
        for y in range(tile.bottom, tile.top):
            n_placed += RNG.random_draw(A*(rho**x))

    particles = np.ndarray((n_placed, PARTICLE_FIELDS), dtype=np.float64)
    pi = 0
    for x in range(tile.left, tile.right):
        start_index = tile.bottom+x*L
        RNG.jump(2*start_index, 0)
        for y in range(tile.bottom, tile.top):
            n_tile_particles = RNG.random_draw(A*(rho**x))
            for p in range(n_tile_particles):
                this_particle = particles[pi]
                this_particle[PARTICLE_X] = x + REL_X
                this_particle[PARTICLE_Y] = y + REL_Y
                this_particle[PARTICLE_K] = k
                this_particle[PARTICLE_M] = m
                pi += 1
    return particles

@njit
def compute_coulomb(x_dist: float, y_dist: float, q1: float, q2: float, forces):

    r2 = x_dist * x_dist + y_dist * y_dist
    r = math.sqrt(r2)
    f_coulomb = q1 * q2 / r2
    forces[0] = f_coulomb * x_dist/r  # f_coulomb * cos_theta
    forces[1] = f_coulomb * y_dist/r  # f_coulomb * sin_theta

    return 0

@njit
def compute_total_force(p: np.ndarray, tile: BoundingBox, grid: np.ndarray, forces):
    temp_forces = np.empty(2, dtype=np.float64)

    n_rows = tile.top-tile.bottom+1

    # Coordinates of the cell containing the particle
    y = math.floor(p[PARTICLE_Y])
    x = math.floor(p[PARTICLE_X])
    x_sav=x

    rel_x = p[PARTICLE_X] - x
    rel_y = p[PARTICLE_Y] - y

    x = x - tile.left
    y = y - tile.bottom

    compute_coulomb(rel_x, rel_y, p[PARTICLE_Q], grid[x,y], temp_forces)

    tmp_res_x = temp_forces[0]
    tmp_res_y = temp_forces[1]

    # Coulomb force from bottom-left charge
    compute_coulomb(rel_x, 1.0-rel_y, p[PARTICLE_Q], grid[x,y+1], temp_forces)
    tmp_res_x += temp_forces[0]
    tmp_res_y -= temp_forces[1]

    # Coulomb force from top-right charge
    compute_coulomb(1.0-rel_x, rel_y, p[PARTICLE_Q], grid[x+1,y], temp_forces)
    tmp_res_x -= temp_forces[0]
    tmp_res_y += temp_forces[1]

    # Coulomb force from bottom-right charge
    compute_coulomb(1.0-rel_x, 1.0-rel_y, p[PARTICLE_Q], grid[x+1,y+1], temp_forces)
    tmp_res_x -= temp_forces[0]
    tmp_res_y -= temp_forces[1]

    forces[0] = tmp_res_x
    forces[1] = tmp_res_y

@njit
def update_particle(p: np.ndarray, forces: np.ndarray, L: int):
    ax = forces[0] * MASS_INV
    ay = forces[1] * MASS_INV

    p[PARTICLE_X] = (p[PARTICLE_X] + p[PARTICLE_VX]*DT + 0.5*ax*DT*DT + L) % L
    p[PARTICLE_Y] = (p[PARTICLE_Y] + p[PARTICLE_VY]*DT + 0.5*ay*DT*DT + L) % L

    p[PARTICLE_VX] = ax * DT
    p[PARTICLE_VY] = ay * DT

@njit
def find_owner_simple(p: np.ndarray, width: int, height: int,
                      num_procsx: int, icrit: int, jcrit: int,
                      ileftover: int, jleftover: int
                      ):
    x = math.floor(p[PARTICLE_X])
    y = math.floor(p[PARTICLE_Y])

    IDx = x // width
    IDy = y // height

    return IDy * num_procsx + IDx

@njit
def find_owner_general(p: np.ndarray, width: int, height: int,
                      num_procsx: int, icrit: int, jcrit: int,
                      ileftover: int, jleftover: int
                      ):
    x = math.floor(p[PARTICLE_X])
    y = math.floor(p[PARTICLE_Y])
    if x < icrit:
        idx = x // (width+1)
    else:
        idx = ileftover + (x-icrit) // width

    if y < jcrit:
        idy = y // (height+1)
    else:
        idy = jleftover + (y-jcrit) // height

    return idy * num_procsx + idx

@njit
def verify_particle(p: np.ndarray, L: float, iterations: int):
    x_final = p[PARTICLE_X0] + (iterations+1) * (2.0*p[PARTICLE_K]+1)
    y_final = p[PARTICLE_Y0] + (iterations+1) * p[PARTICLE_M]

    x_periodic = (x_final % L) if x_final >= 0.0 else L + (x_final % L)
    y_periodic = (y_final % L) if (y_final >= 0.0) else L + (y_final % L)

    return not (abs(p[PARTICLE_X] - x_periodic) > epsilon \
                or abs(p[PARTICLE_Y] - y_periodic) > epsilon)

def add_particle_to_buffer(p: np.ndarray, array: array):
    array.extend(p)

def attach_received_particles(my_particles: np.ndarray,
                              received_particles: np.ndarray
                              ):
    n_received = len(received_particles) // PARTICLE_FIELDS
    received = received_particles.reshape((n_received,
                                           PARTICLE_FIELDS)
                                           )
    return np.concatenate([my_particles, received])


def bad_patch(patch: BoundingBox, patch_contain: BoundingBox):
    if patch.left >= patch.right or patch.bottom >= patch.top:
        return 1
    if patch_contain:
        if patch.left < patch_contain.left or patch.right >= patch_contain.right:
            return 2
        if patch.bottom < patch_contain.bottom or patch.top >= patch_contain.top:
            return 3
    return 0

def contain(x: int, y: int, patch: BoundingBox):
    conds = [x < patch.left, x > patch.right,
             y < patch.bottom, y > patch.top
             ]
    return not any(conds)

def resize_buffer(buf: array, counts: int):
    if len(buf) < counts:
        buf.extend(bytearray(counts-len(buf)))


