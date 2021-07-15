#!/usr/bin/env python3
import sys
from charm4py import *
import numpy
from enum import Enum
import time
import kernels

COMM_TIME, COMP_TIME, ITER_TIME = range(3)

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = r/fac1
            break;
    return fac1, fac2

class Directions(Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

class Cell(Chare):
    def __init__(self):
        self.X, self.Y = self.thisIndex
        my_blocksize = (n//x)*(m//y)
        ghost_size = ((n//x) * 2) + ((m//y) * 2)
        kernels.set_block_params(m//y, n//x)

        self.width = m//y
        self.height = n//x

        self.T = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
        self.newT = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
        kernels.enforce_BC(self.T)

    @coro
    def run(self, done_future):
        neighbors = list()
        # information for compute, comm., and total time for all iterations
        # We consider communication time the time it takes to pack, send, and
        # unpack ghosts.
        # Computation is just time spent in the kernel and enforcing BC
        timing_info = numpy.zeros((warmup + iterations, 3), dtype=numpy.float64)

        if self.Y > 0:
            top_proxy = self.thisProxy[(self.X, self.Y-1)]
            top_nbr = Channel(self, top_proxy)
            top_nbr.dir = Directions.TOP
            neighbors.append(top_nbr)

        if self.Y < y-1:
            bot_proxy = self.thisProxy[(self.X, self.Y+1)]
            bot_nbr = Channel(self, bot_proxy)
            bot_nbr.dir = Directions.BOTTOM
            neighbors.append(bot_nbr)

        if self.X > 0:
            left_proxy = self.thisProxy[(self.X-1, self.Y)]
            left_nbr = Channel(self, left_proxy)
            left_nbr.dir = Directions.LEFT
            neighbors.append(left_nbr)

        if self.X < x-1:
            right_proxy = self.thisProxy[(self.X+1, self.Y)]
            right_nbr = Channel(self, right_proxy)
            right_nbr.dir = Directions.RIGHT
            neighbors.append(right_nbr)

        if np > 1:
            top_buf_out = numpy.zeros(self.width)
            bot_buf_out = numpy.zeros(self.width)

            right_buf_out = numpy.zeros(self.height)
            left_buf_out = numpy.zeros(self.height)

        for i in range(warmup + iterations):
            if i == warmup:
                t0 = time.perf_counter()

            iter_start = time.perf_counter_ns()
            comm_start = iter_start
            if self.Y > 0:
                kernels.pack_top(self.T, top_buf_out)
                top_nbr.send(top_buf_out)

            if self.Y < y-1:
                kernels.pack_bottom(self.T, bot_buf_out)
                bot_nbr.send(bot_buf_out)

            if self.X < x-1:
                kernels.pack_right(self.T, right_buf_out)
                right_nbr.send(right_buf_out)

            if self.X > 0:
                kernels.pack_left(self.T, left_buf_out)
                left_nbr.send(left_buf_out)

            for ready_ch in charm.iwait(neighbors):
                input_data = ready_ch.recv()
                if ready_ch.dir == Directions.TOP:
                    kernels.unpack_top(self.T, input_data)

                elif ready_ch.dir == Directions.BOTTOM:
                    kernels.unpack_bottom(self.T, input_data)

                elif ready_ch.dir == Directions.LEFT:
                    kernels.unpack_left(self.T, input_data)

                elif ready_ch.dir == Directions.RIGHT:
                    kernels.unpack_right(self.T, input_data)

            comm_end = time.perf_counter_ns()
            comp_start = comm_end

            # Apply the stencil operator
            kernels.compute(self.newT, self.T)
            self.newT, self.T = self.T, self.newT
            kernels.enforce_BC(self.T)
            iter_end = time.perf_counter_ns()
            comp_end = iter_end

            timing_info[i][COMM_TIME] = comm_end - comm_start
            timing_info[i][COMP_TIME] = comp_end - comp_start
            timing_info[i][ITER_TIME] = iter_end - iter_start

        self.local_time = time.perf_counter() - t0
        self.reduce(self.thisProxy[(0, 0)].gather_timing,
                    timing_info,
                    Reducer.gather
                    )

        if self.thisIndex == (0, 0):
            self.done_future = done_future
        else:
            self.reduce(done_future)

        if self.thisIndex == (0, 0):
            print(f"Elapsed: {self.local_time}")


    def gather_timing(self, gathered):
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"{output_filepath}/{dt_string}_n{n}_m{n}_np{np}_i{iterations}_w{warmup}_charm.csv"
        header = "Process,Iteration,Iteration Time,Communication Time,Computation Time"
        with open(output_filename, 'w') as output_file:
            output_file.write(f'#{" ".join(sys.argv)}\n')
            output_file.write(header + '\n')
            for pnum, timing in enumerate(gathered):
                for inum, iter in enumerate(timing):
                    in_seconds = iter/1e9
                    comm_time = in_seconds[COMM_TIME]
                    comp_time = in_seconds[COMP_TIME]
                    iter_time = in_seconds[ITER_TIME]
                    output_tuple = (pnum, inum+1, iter_time, comm_time, comp_time)
                    output_file.write(','.join(map(str, output_tuple)) + '\n')
        self.reduce(self.done_future)

def main(args):
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python Charm/Numpy  Stencil execution on 2D grid')

    if len(sys.argv) < 3:
        print('argument count = ', len(sys.argv))
        charm.abort("Usage: ./stencil <# chares> <# iterations> "
                    "[<array dimension> or <array dimension X> <array dimension Y>]"
                    )

    np = int(sys.argv[1])
    if np < 1:
        charm.abort("ERROR: num_chares must be >= 1")
    iterations = int(sys.argv[2])
    if iterations < 1:
        charm.abort("ERROR: iterations must be >= 1")

    n = int(sys.argv[3])
    if len(sys.argv) > 4:
        m = int(sys.argv[4])
    else:
        m = n

    nsquare = n * m
    x, y = factor(np)
    x, y = int(x), int(y)

    if nsquare < np:
        charm.abort(f"ERROR: grid size {nsquare}, must be at least # ranks: {np}")
    if n % x:
        charm.abort(f"ERROR: grid size {n} does not evenly divide the number of chares in the x dimension {x}")
    if m % y:
        charm.abort(f"ERROR: grid size {m} does not evenly divide the number of chares in the y dimension {y}")

    params = {'x': x, 'y': y,
              'np': np, 'iterations': iterations,
              'n': n, 'm': m,
              'warmup':10,
              'output_filepath': '.'
            }

    charm.thisProxy.updateGlobals(params, awaitable=True).get()
    cells = Array(Cell, (x, y))

    print('Number of chares            = ', np)
    print('Number of processors        = ', charm.numPes())
    print('Grid shape (x,y)            = ', x, y)
    print('Problem Domain (x,y)        = ', n, m)
    print('Number of iterations        = ', iterations)
    print('Number of warmup iterations = ', warmup)

    done_fut = Future()
    tstart = time.perf_counter()
    cells.run(done_fut)
    done_fut.get()
    tend = time.perf_counter()
    print(f"Elapsed: {tend-tstart}")
    charm.exit()


charm.start(main)
