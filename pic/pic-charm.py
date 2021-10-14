import time
from definitions import *
from kernels import *
from random_draw import *
from array import array
from charm4py import *
import numpy as np
import sys


class Cell(Chare):
    def __init__(self, done_future):
        self.done_future = done_future
        rank = self.thisIndex[0]
        my_idx = rank % num_charesx
        my_idy = rank // num_charesx

        icrit = 0
        jcrit = 0
        width = L // num_charesx
        if width < 2*k:
            if rank == 0:
                print(f"k-value too large: {k}, must be no greater than: {width//2}")
                charm.exit()

        ileftover = L % num_charesx
        if my_idx < ileftover:
            istart = (width+1) * my_idx
            iend = istart + width + 1
        else:
            istart = (width+1) * ileftover + width * (my_idx - ileftover)
            iend = istart + width

        icrit = (width + 1) * ileftover

        height = L // num_charesy
        if height < m:
            if rank == 0:
                print(f"m-value too large: {m}, must be no greater than {height}")
                charm.exit()

        jleftover = L % num_charesy
        if my_idy < jleftover:
            jstart = (height+1) * my_idy
            jend = jstart + height + 1
        else:
            jstart = (height+1) * jleftover + height * (my_idy-jleftover)
            jend = jstart + height

        jcrit = (height+1) * jleftover

        if icrit == 0 and jcrit == 0:
            self.find_owner = find_owner_simple
        else:
            self.find_owner = find_owner_general

        my_tile = BoundingBox(istart, iend, jstart, jend)

        nbr = [0] * 8
        nbr[0] = rank + num_charesx - 1 if my_idx == 0 else rank - 1
        nbr[1] = rank - num_charesx + 1 if my_idx == num_charesx-1 else rank + 1
        nbr[2] = rank + num_charesx - num_chares if my_idy == num_charesy-1 else rank + num_charesx
        nbr[3] = rank - num_charesx + num_chares if my_idy == 0 else rank - num_charesx
        nbr[4] = nbr[0] + num_charesx - num_chares if my_idy == num_charesy-1 else nbr[0] + num_charesx
        nbr[5] = nbr[1] + num_charesx - num_chares if my_idy == num_charesy-1 else nbr[1] + num_charesx
        nbr[6] = nbr[0] - num_charesx + num_chares if my_idy == 0 else nbr[0] - num_charesx
        nbr[7] = nbr[1] - num_charesx + num_chares if my_idy == 0 else nbr[1] - num_charesx


        grid = initialize_grid(my_tile)
        particles = initialize_geometric(n, L, rho, my_tile, k, m)
        num_particles = len(particles)

        self.rank = rank
        self.nbr = nbr
        self.grid = grid
        self.particles = particles
        self.num_particles = num_particles
        self.my_tile = my_tile
        self.crit = (icrit, jcrit)
        self.leftover = (ileftover, jleftover)
        self.ibounds = (istart, iend)
        self.jbounds = (jstart, jend)
        self.width = width
        self.height = height
        self.neighbors = [Channel(self,
                                  self.thisProxy[nbr_idx]
                                  ) for nbr_idx in self.nbr
                          ]
        self.reduce(self.thisProxy.finish_initializing, num_particles, Reducer.gather)
        self.iter = 0
        if verbose:
            print(f"Chare {rank} has {num_particles} particles.")

    def validate(self, values):
        print("Validation values: ", values, type(values), len(values))
        total_incorrect, id_checksum, nparts = values

        tp = self.total_particles
        num_particles_checksum = (tp)*(tp+1) // 2
        if total_incorrect:
            print(f"There are {total_incorrect} miscalculated particle locations.")
        else:
            if id_checksum != num_particles_checksum:
                print("Particle checksum incorrect.")
            else:
                print("Solution validates.")

    @coro
    def run(self):
        rank = self.rank
        icrit, jcrit = self.crit
        ileftover, jleftover = self.leftover

        iter = self.iter
        if iter < 2:
            self.allreduce().get()
            self.sim_start = wtime()
        start_particles = len(self.particles)
        # we need the size of these buffers to be exactly the
        # number of particles sent/received
        # (unless we don't, should check later)
        my_buf = array('d')

        send_bufs = [array('d') for _ in range(len(self.nbr))]
        forces = np.ndarray((2,), dtype=np.float64)
        for p in self.particles:
            compute_total_force(p, self.my_tile, self.grid, forces)
            update_particle(p, forces, L)

            owner = self.find_owner(p, self.width, self.height, num_charesx,
                                    icrit, jcrit, ileftover, jleftover
                                    )

            if owner == self.rank:
                add_particle_to_buffer(p, my_buf)
            elif owner == self.nbr[0]:
                add_particle_to_buffer(p, send_bufs[0])
            elif owner == self.nbr[1]:
                add_particle_to_buffer(p, send_bufs[1])
            elif owner == self.nbr[2]:
                add_particle_to_buffer(p, send_bufs[2])
            elif owner == self.nbr[3]:
                add_particle_to_buffer(p, send_bufs[3])
            elif owner == self.nbr[4]:
                add_particle_to_buffer(p, send_bufs[4])
            elif owner == self.nbr[5]:
                add_particle_to_buffer(p, send_bufs[5])
            elif owner == self.nbr[6]:
                add_particle_to_buffer(p, send_bufs[6])
            elif owner == self.nbr[7]:
                add_particle_to_buffer(p, send_bufs[7])
            else:
                print(f"{rank}: Could not find neighbor owner of particle "
                      f"{p[PARTICLE_ID]} in tile {owner}, "
                      f" my neighbors are {self.nbr}, particle idx is {p[PARTICLE_ID]}, total particles: {len(self.particles)}, particle is {p}"
                      )
                sys.exit()

        num_particles = len(my_buf) // PARTICLE_FIELDS
        self.particles = np.frombuffer(my_buf, dtype=np.float64)
        self.particles = self.particles.reshape((num_particles, PARTICLE_FIELDS))

        for nbr_idx, neighbor in enumerate(self.neighbors):
            neighbor.send(send_bufs[nbr_idx])

        for ch in charm.iwait(self.neighbors):
            p_recv = np.frombuffer(ch.recv(), dtype=np.float64)
            self.particles = attach_received_particles(self.particles, p_recv)

        end_particles = len(self.particles)

        self.num_particles = len(self.particles)
        if rank == 0 and verbose:
            print(f"Iteration {iter} complete in {(iter_end - iter_start)}s.")

        self.iter += 1

        if 0 < self.iter < iterations + 1 and self.iter % migration_delay == 0:
            self.AtSync()
            return

        if self.iter < iterations + 1:
            self.run()
        else:
            self.allreduce().get()
            if self.rank == 0:
                sim_elapsed = wtime() - self.sim_start
                print(f"Sim elapsed: {sim_elapsed}")

            n_incorrect = 0
            id_checksum = 0
            for p in self.particles:
                n_incorrect += int(verify_particle(p, L, iterations + 1))
                id_checksum += int(p[PARTICLE_ID])

            self.reduce(self.thisProxy[0].validate,
                        [n_incorrect, id_checksum],
                        Reducer.sum
                        )
            self.reduce(self.done_future)

    def resumeFromSync(self):
        self.thisProxy[self.thisIndex].run()

    def finish_initializing(self, particle_counts):
        n_prefix = sum(particle_counts[0:self.rank+1])
        finish_particle_initialization(self.particles, n_prefix)
        total_particles = sum(particle_counts)

        if self.rank == 0:
            self.total_particles = total_particles
            print(f"Total particles in the simulation: {total_particles}")

        self.reduce(self.thisProxy.run)

    def validate(self, values):
        total_incorrect, id_checksum = values

        tp = self.total_particles
        num_particles_checksum = (tp)*(tp+1) // 2
        if total_incorrect:
            print(f"There are {total_incorrect} miscalculated particle locations.")
        else:
            if id_checksum != num_particles_checksum:
                print(f"Particle checksum incorrect. {num_particles_checksum} {id_checksum}")
            else:
                print("Solution validates.")

def main(args):
    sim_params = parse_args(args)
    validate_args(sim_params)

    num_chares = sim_params.num_chares
    iterations = sim_params.iterations
    period = sim_params.period
    migration_delay = sim_params.migration_delay
    n = sim_params.num_particles
    L = sim_params.grid_size
    rho = float(sim_params.init_parameters[0])
    k = sim_params.charge_increment
    m = sim_params.velocity_y


    num_charesx = 0
    num_charesy = 0

    num_charesx = int(math.sqrt(num_chares+1))

    while num_charesx > 0:
        if not (num_chares % num_charesx):
            num_charesy = num_chares // num_charesx
            break
        num_charesx -= 1

    print(num_charesx, num_charesy)
    params = {'num_chares': num_chares, 'iterations': iterations,
              'period': period, 'migration_delay': migration_delay,
              'n': n, 'L': L, 'rho': rho, 'k': k,
              'm': m, 'num_charesx': num_charesx,
              'num_charesy': num_charesy,
              'verbose': sim_params.verbose,
              'output': sim_params.output,
              'add_datetime': sim_params.add_datetime
              }

    done_future = Future()
    charm.thisProxy.updateGlobals(params, awaitable=True).get()
    chares = Array(Cell, num_chares, args=[done_future],useAtSync=True)
    done_future.get()
    charm.exit()

if __name__ == '__main__':
    charm.start(main)
