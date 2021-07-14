#!/usr/bin/env python3
import sys
from charm4py import *
import numpy
from enum import Enum
import time
import kernels

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
        my_blocksize = (n//x)*(n//y)
        ghost_size = ((n//x) * 2) + ((n//y) * 2)
        kernels.set_block_params(n//x, n//y)

        self.width = n//x
        self.height = n//y

        self.T = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
        self.newT = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)

    @coro
    def run(self, done_future):
        neighbors = list()

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
            # Apply the stencil operator
            kernels.compute(self.newT, self.T)
            self.newT, self.T = self.T, self.newT
            kernels.enforce_BC(self.T)

        local_time = time.perf_counter() - t0
        # assert numpy.allclose(self.newT, self.T)
        if self.thisIndex == (0,0):
            print(f"Elapsed: {local_time}")

        self.reduce(done_future)

def main(args):
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Python Charm/Numpy  Stencil execution on 2D grid')

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('argument count = ', len(sys.argv))
        charm.exit("Usage: ./stencil <# chares> <# iterations> <array dimension>")

    np = int(sys.argv[1])
    if np < 1:
        charm.exit("ERROR: num_chares must be >= 1")
    iterations = int(sys.argv[2])
    if iterations < 1:
        charm.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[3])
    nsquare = n * n
    if nsquare < np:
        charm.abort(f"ERROR: grid size {nsquare}, must be at least # ranks: {np}")
    if n % np:
        charm.abort(f"ERROR: grid size {n} does not evenly divide the number of chares {np}")

    x, y = factor(np)
    x, y = int(x), int(y)


    params = {'x': x, 'y': y,
              'np': np, 'iterations': iterations,
              'n': n,
              'warmup':10
            }

    charm.thisProxy.updateGlobals(params, awaitable=True).get()
    cells = Array(Cell, (x, y))

    print('Number of chares          = ', np)
    print('Number of processors      = ', charm.numPes())
    print('Grid shape (x,y)          = ', x, y)
    print('Number of iterations      = ', iterations)

    done_fut = Future()
    tstart = time.perf_counter()
    cells.run(done_fut)
    done_fut.get()
    tend = time.perf_counter()
    print(f"Elapsed: {tend-tstart}")
    charm.exit()


charm.start(main)
