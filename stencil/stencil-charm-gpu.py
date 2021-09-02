#!/usr/bin/env python3
import sys
from charm4py import *
import numpy
from enum import Enum
import time
import gpu_kernels as kernels
from array import array

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

        width, height = self.width, self.height

        top_buf_out_h = numpy.zeros(width)
        top_buf_in_h = numpy.zeros(width)
        bot_buf_out_h = numpy.zeros(width)
        bot_buf_in_h = numpy.zeros(width)

        self.top_buf_out = cuda.to_device(top_buf_out_h)
        self.top_buf_in = cuda.to_device(top_buf_in_h)
        self.bot_buf_out = cuda.to_device(bot_buf_out_h)
        self.pot_buf_in = cuda.to_device(pot_buf_in_h)

        right_buf_out_h = numpy.zeros(height)
        right_buf_in_h = numpy.zeros(height)
        left_buf_out_h = numpy.zeros(height)
        left_buf_in_h = numpy.zeros(height)

        self.right_buf_out = cuda.to_device(right_buf_out_h)
        self.right_buf_in = cuda.to_device(right_buf_in_h)
        self.left_buf_out = cuda.to_device(left_buf_out_h)
        self.left_buf_in = cuda.to_device(left_buf_in_h)

        kernels.enforce_BC(self.T)

    @coro
    def run(self, done_future):
        neighbors = list()
        # information for compute, comm., and total time for all iterations
        # We consider communication time the time it takes to pack, send, and
        # unpack ghosts.
        # Computation is just time spent in the kernel and enforcing BC
        timing_info = numpy.zeros((warmup + iterations, 3), dtype=numpy.float64)
        empty = lambda x: [0] * x
        get_address = lambda x: x.__cuda_array_interface__['data'][0]

        if self.Y > 0:
            top_proxy = self.thisProxy[(self.X, self.Y-1)]
            top_nbr = Channel(self, top_proxy)
            top_nbr.dir = Directions.TOP

            address = get_address(self.top_buf_in)
            address = get_address(self.top_buf_out)
            size = len(self.top_buf_in)
            top_nbr.dev_addr_out = array.array('L', [address_out])
            top_nbr.dev_size_out = array.array('i', [size])

            top_nbr.dev_addr_in = array.array('L', [address_in])
            top_nbr.dev_size_in = array.array('i', [size])

            self.TOP = top_nbr
            neighbors.append(top_nbr)

        if self.Y < y-1:
            bot_proxy = self.thisProxy[(self.X, self.Y+1)]
            bot_nbr = Channel(self, bot_proxy)
            bot_nbr.dir = Directions.BOTTOM
            address_in = get_address(self.bot_buf_in)
            address_out = get_address(self.bot_buf_out)
            size = len(self.bot_buf_in)
            bot_nbr.dev_addr_out = array.array('L', [address_out])
            bot_nbr.dev_size_out = array.array('i', [size])

            bot_nbr.dev_addr_in = array.array('L', [address_in])
            bot_nbr.dev_size_in = array.array('i', [size])

            self.BOTTOM = bottom_nbr
            neighbors.append(bot_nbr)

        if self.X > 0:
            left_proxy = self.thisProxy[(self.X-1, self.Y)]
            left_nbr = Channel(self, left_proxy)
            left_nbr.dir = Directions.LEFT

            address_in = get_address(self.left_buf_in)
            address_out = get_address(self.left_buf_out)
            size = len(self.left_buf_in)
            left_nbr.dev_addr_out = array.array('L', [address_out])
            left_nbr.dev_size_out = array.array('i', [size])

            left_nbr.dev_addr_in = array.array('L', [address_in])
            left_nbr.dev_size_in = array.array('i', [size])

            self.LEFT = left_nbr
            neighbors.append(left_nbr)

        if self.X < x-1:
            right_proxy = self.thisProxy[(self.X+1, self.Y)]
            right_nbr = Channel(self, right_proxy)
            right_nbr.dir = Directions.RIGHT

            address_in = get_address(self.right_buf_in)
            address_out = get_address(self.right_buf_out)
            size = len(self.right_buf_in)

            right_nbr.dev_addr_out = array.array('L', [address_out])
            right_nbr.dev_size_out = array.array('i', [size])

            right_nbr.dev_addr_in = array.array('L', [address_in])
            right_nbr.dev_size_in = array.array('i', [size])

            self.RIGHT = right_nbr
            neighbors.append(right_nbr)

        for i in range(warmup + iterations):
            if i == warmup:
                t0 = time.perf_counter()

            iter_start = time.perf_counter_ns()
            comm_start = iter_start
            if self.Y > 0:
                kernels.pack_top(self.T, self.top_buf_out)
                top_nbr.send(src_ptrs=self.TOP.dev_addr_out,
                             src_sizes=self.TOP.dev_size_out
                             )
            if self.Y < y-1:
                kernels.pack_bottom(self.T, self.bot_buf_out)
                bot_nbr.send(src_ptrs=self.BOTTOM.dev_addr_out,
                             src_sizes=self.BOTTOM.dev_size_out
                             )

            if self.X < x-1:
                kernels.pack_right(self.T, self.right_buf_out)
                right_nbr.send(src_ptrs=self.RIGHT.dev_addr_out,
                               src_sizes=self.RIGHT.dev_size_out
                               )

            if self.X > 0:
                kernels.pack_left(self.T, left_buf_out)
                left_nbr.send(src_ptrs=self.LEFT.dev_addr_out,
                              src_sizes=self.LEFT.dev_size_out
                              )

            charm.iwait_map(self.receive_ghost, neighbors)

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

    def receive_ghost(self, channel):
        channel.recv(post_addresses=channel.dev_addr_in,
                     post_sizes=channel.dev_size_in
                     )

        if channel.dir == Directions.TOP:
            kernels.unpack_top(self.T, self.top_buf_in)

        elif channel.dir == Directions.BOTTOM:
            kernels.unpack_bottom(self.T, self.bot_buf_in)

        elif channel.dir == Directions.LEFT:
            kernels.unpack_left(self.T, self.left_buf_in)

        elif channel.dir == Directions.RIGHT:
            kernels.unpack_right(self.T, self.right_buf_in)


    def gather_timing(self, gathered):
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"{output_filepath}/{dt_string}_n{n}_m{n}_np{np}_i{iterations}_w{warmup}_charm.csv"
        header = "Process,Num processes,Iteration,Iteration Time,Communication Time,Computation Time"
        with open(output_filename, 'w') as output_file:
            output_file.write(f'#{" ".join(sys.argv)}\n')
            output_file.write(header + '\n')
            for pnum, timing in enumerate(gathered):
                for inum, iter in enumerate(timing):
                    in_seconds = iter/1e9
                    comm_time = in_seconds[COMM_TIME]
                    comp_time = in_seconds[COMP_TIME]
                    iter_time = in_seconds[ITER_TIME]
                    output_tuple = (pnum, np, inum+1, iter_time, comm_time, comp_time)
                    output_file.write(','.join(map(str, output_tuple)) + '\n')
        self.reduce(self.done_future)

def main(args):
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python Charm/Numpy  Stencil execution on 2D grid')

    if len(args) < 3:
        print('argument count = ', len(args))
        charm.abort("Usage: ./stencil <# chares> <# iterations> "
                    "[<array dimension> or <array dimension X> <array dimension Y>]"
                    )

    np = int(args[1])
    if np < 1:
        charm.abort("ERROR: num_chares must be >= 1")
    iterations = int(args[2])
    if iterations < 1:
        charm.abort("ERROR: iterations must be >= 1")

    n = int(args[3])
    if len(args) > 4:
        m = int(args[4])
    else:
        m = n

    output_filepath = '.'
    try:
        arg_int = int(args[-1])
    except:
        output_filepath = args[-1]

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
              'output_filepath': output_filepath
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
    print(f"Elapsed (total): {tend-tstart}")
    charm.exit()


charm.start(main)
