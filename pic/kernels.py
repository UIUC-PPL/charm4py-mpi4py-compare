from definitions import *
import math
import random
from random_draw import *

RNG = LCG()

def initialize_grid(tile: BoundingBox):
    n_columns = tile.right-tile.left+1
    n_rows = tile.top-tile.bottom+1
    grid = np.ndarray((n_columns, n_rows), dtype=np.float64)

    for y in range(tile.bottom, tile.top+1):
        for x in range(tile.left, tile.right+1):
            grid[y-tile.bottom, x-tile.left] = Q if x % 2 == 0 else -Q
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
