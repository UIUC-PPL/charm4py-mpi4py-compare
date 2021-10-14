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
    my_idy = rank // num_procsx


    width = L // num_procsx
    if width < 2*k:
        if rank == 0:
            print(f"k-value too large: {k}, must be no greater than: {width//2}")
        sys.exit()

    ileftover = L % num_procsx
    if my_idx < ileftover:
        istart = (width+1) * my_idx
        iend = istart + width + 1
    else:
        istart = (width+1) * ileftover + width * (my_idx - ileftover)
        iend = istart + width

    icrit = (width + 1) * ileftover

    height = L // num_procsy
    if height < m:
        if rank == 0:
            print(f"m-value too large: {m}, must be no greater than {height}")
        sys.exit()

    jleftover = L % num_procsy
    if my_idy < jleftover:
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
    nbr[0] = rank + num_procsx - 1 if my_idx == 0 else rank - 1
    nbr[1] = rank - num_procsx + 1 if my_idx == num_procsx-1 else rank + 1
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

    send_counts = np.ndarray((8, 1), dtype=np.int32)
    recv_counts = np.ndarray((8, 1), dtype=np.int32)

    # total time, compute time, communicate time, (total) simulation time,
    # start particles, end particles
    timers = np.ndarray((iterations+1, 6), dtype=np.float64)

    # perform the simulation
    for iter in range(iterations + 1):
        if iter == 1:
            comm.barrier()
        if iter < 2:
            # include some data for warmup iter, but
            # dont' include in total
            sim_start = wtime()

        iter_start = wtime()
        comp_start = iter_start
        start_particles = len(particles)
        # we need the size of these buffers to be exactly the
        # number of particles sent/received
        # (unless we don't, should check later)
        my_buf = array('d')

        send_bufs = [array('d') for _ in range(len(nbr))]
        recv_bufs = [array('d') for _ in range(len(nbr))]
        forces = np.ndarray((2,), dtype=np.float64)
        for p in particles:
            compute_total_force(p, my_tile, grid, forces)
            update_particle(p, forces, L)

            owner = find_owner(p, width, height, num_procsx,
                               icrit, jcrit, ileftover, jleftover
                               )

            if owner == rank:
                add_particle_to_buffer(p, my_buf)
            elif owner == nbr[0]:
                add_particle_to_buffer(p, send_bufs[0])
            elif owner == nbr[1]:
                add_particle_to_buffer(p, send_bufs[1])
            elif owner == nbr[2]:
                add_particle_to_buffer(p, send_bufs[2])
            elif owner == nbr[3]:
                add_particle_to_buffer(p, send_bufs[3])
            elif owner == nbr[4]:
                add_particle_to_buffer(p, send_bufs[4])
            elif owner == nbr[5]:
                add_particle_to_buffer(p, send_bufs[5])
            elif owner == nbr[6]:
                add_particle_to_buffer(p, send_bufs[6])
            elif owner == nbr[7]:
                add_particle_to_buffer(p, send_bufs[7])
            else:
                print(f"{rank}: Could not find neighbor owner of particle "
                      f"{p[PARTICLE_ID]} in tile {owner}, "
                      f" my neighbors are {nbr}, particle idx is {idx}, total particles: {len(particles)}, particle is {p}"
                      )
                sys.exit()
        comp_end = wtime()
        comm_start = wtime()

        requests = [MPI.REQUEST_NULL for i in range(16)]
        for i in range(8):
            send_counts[i] = len(send_bufs[i])
            r2 = comm.Irecv(recv_counts[i], source=nbr[i], tag=0)
            r1 = comm.Isend(send_counts[i], dest=nbr[i], tag=0)
            requests[i] = r1
            requests[i+8] = r2
        MPI.Request.Waitall(requests)

        for i in range(8):
            recv_bufs[i] = np.ndarray(recv_counts[i], dtype=np.float64)

        requests = [MPI.REQUEST_NULL for i in range(16)]
        for i in range(8):
            r2 = comm.Irecv([recv_bufs[i], recv_counts[i], MPI.DOUBLE],
                            source=nbr[i], tag=0
                            )
            r1 = comm.Isend([send_bufs[i], len(send_bufs[i]), MPI.DOUBLE],
                            dest=nbr[i], tag=0
                            )
            requests[i] = r1
            requests[i+8] = r2

        num_particles = len(my_buf) // PARTICLE_FIELDS
        particles = np.frombuffer(my_buf, dtype=np.float64)
        particles = particles.reshape((num_particles, PARTICLE_FIELDS))
        MPI.Request.Waitall(requests)

        for i in range(8):
            # Maybe it's better to first combine all received particles?
            # Can we somehow receive all particles into the same buffer?
            particles = attach_received_particles(particles, recv_bufs[i])

        comm_end = wtime()
        iter_end = comm_end
        end_particles = len(particles)

        timers[iter][ELAPSED_TIME] = iter_end - sim_start
        timers[iter][ITER_TIME] = iter_end - iter_start
        timers[iter][COMP_TIME] = comp_end - comp_start
        timers[iter][COMM_TIME] = comm_end - comm_start
        timers[iter][START_PARTICLES] = start_particles
        timers[iter][END_PARTICLES] = end_particles

        num_particles = len(particles)
        if rank == 0 and sim_params.verbose:
            print(f"Iteration {iter} complete in {iter_end - iter_start}s.")
    comm.barrier()
    sim_end = wtime()
    sim_elapsed = sim_end - sim_start
    if rank == 0:
        print(f"Sim elapsed: {sim_elapsed}")
    n_incorrect = 0
    id_checksum = 0
    for p in particles:
        n_incorrect += int(verify_particle(p, L, iterations + 1))
        id_checksum += int(p[PARTICLE_ID])

    reduction_vals = np.ndarray(2, dtype=np.int64)
    reduction_vals[0] = n_incorrect
    reduction_vals[1] = id_checksum
    totals = np.zeros(2, dtype=np.int64)
    comm.Reduce([reduction_vals, MPI.LONG_LONG],
                [totals, MPI.LONG_LONG],
                op=MPI.SUM, root=0
                )
    all_timers = comm.gather(timers, root=0)
    # write timers to file (if provided)
    if rank == 0:
        total_incorrect, id_checksum = totals
        num_particles_checksum = (total_particles)*(total_particles+1) // 2
        if total_incorrect:
            print(f"There are {total_incorrect} miscalculated particle locations.")
        else:
            if id_checksum != num_particles_checksum:
                print("Particle checksum incorrect.")
            else:
                print("Solution validates.")
        if sim_params.output:
            if sim_params.add_datetime:
                dt_str = get_datetime_str()
            else:
                dt_str = None
            write_output(sim_params.output, all_timers, prefix=dt_str)

    exit()


if __name__ == '__main__':
    main()
