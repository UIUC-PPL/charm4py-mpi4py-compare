from definitions import *
from kernels import *
from random_draw import *
from array import array
from mpi4py import MPI
import numpy as np
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    if rank == 0:
        sim_params = parse_args(sys.argv)
        validate_args(sim_params)
    else:
        sim_params = {}
    sim_params = comm.bcast(sim_params, root=0)

    iterations = sim_params.iterations
    period = sim_params.period
    migration_delay = sim_params.migration_delay
    n = sim_params.num_particles
    L = sim_params.grid_size
    rho = float(sim_params.init_parameters[0])
    k = sim_params.charge_increment
    m = sim_params.velocity_y
    grid_patch = BoundingBox(0, L+1, 0, L+1)

    num_procsx = 0
    num_procsy = 0

    num_procsx = int(math.sqrt(num_procs+1))

    while num_procsx > 0:
        if not (num_procs % num_procsx):
            num_procsy = num_procs // num_procsx
            break
        num_procsx -= 1
    my_idx = rank % num_procsx
    my_idy = rank // num_procsy


    width = L / num_procsx
    if width < 2*k:
        if rank == 0:
            print(f"k-value too large: {k}, must be no greater than: {width//2}")
        sys.exit()

    ileftover = L % num_procsx
    if rank < ileftover:
        istart = (width+1) * rank
        iend = istart + width + 1
    else:
        istart = (width+1) * ileftover + width * (rank - ileftover)
        iend = istart + width

    icrit = (width + 1) * ileftover

    height = L / num_procsy
    if height < m:
        if rank == 0:
            print(f"m-value too large: {m}, must be no greater than {height}")
        sys.exit()

    jleftover = L % num_procsy
    if rank < jleftover:
        jstart = (height+1) * my_idy
        jend = jstart + height + 1
    else:
        jstart = (height+1) * jleftover + height * (my_idy-jleftover)
        jend = jstart + height

    jcrit = (height+1) * jleftover

    if icrit == 0 and jcrit == 0:
        find_owner = find_owner_simple
    else:
        # find_owner = find_owner_general
        find_owner = find_owner_general

    my_tile = BoundingBox(istart, iend, jstart, jend)

    nbr = [0] * 8
    nbr[0] = (rank + num_procsx - 1) % num_procsx
    nbr[1] = (rank+1) % num_procsx
    nbr[2] = rank + num_procsx - num_procs if my_idy == num_procsy-1 else rank + num_procsx
    nbr[3] = rank - num_procsx + num_procs if my_idy == 0 else rank - num_procsx
    nbr[4] = nbr[0] + num_procsx - num_procs if my_idy == num_procsy-1 else nbr[0] + num_procsx
    nbr[5] = nbr[1] + num_procsx - num_procs if my_idy == num_procsy-1 else nbr[1] + num_procsx
    nbr[6] = nbr[0] - num_procsx + num_procs if my_idy == 0 else nbr[0] - num_procsx
    nbr[7] = nbr[1] - num_procsx + num_procs if my_idy == 0 else nbr[1] - num_procsx


    grid = initialize_grid(my_tile)
    particles = initialize_geometric(n, L, rho, my_tile, k, m)
    num_particles = len(particles)

    # use Python objects, not on critical path
    n_prefix = comm.scan(num_particles)
    finish_particle_initialization(particles, n_prefix)
    if sim_params.verbose:
        for proc in range(num_procs):
            comm.barrier()
            if proc == rank:
                print(f"Processor {rank} has {num_particles} particles.")
    total_particles = comm.reduce(num_particles, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"Total particles in the simulation: {total_particles}.")
    sys.exit()
    forces = np.ndarray(2, dtype=np.float64)
    compute_total_force(particles[0], box, grid, forces)
    print(particles[0])
    return_val = verify_particle(particles[0], L, 1)
    print(return_val)
    if return_val:
        print("Success!")
    else:
        print("Failure!")
    print("Forces is:", forces)
    exit()


if __name__ == '__main__':
    main()
