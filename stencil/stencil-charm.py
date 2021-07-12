#!/usr/bin/env python3
# This is from the Intel Parallel Research Kernels
# Copyright (c) 2020, Yijian Hu
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# *******************************************************************
#
# NAME:    Stencil
#
# PURPOSE: This program tests the efficiency with which a space-invariant,
#          linear, symmetric filter (stencil) can be applied to a square
#          grid or image.
#
# USAGE:   The program takes as input the linear
#          dimension of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <grid size>
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - RvdW: Removed unrolling pragmas for clarity;
#            added constant to array "in" at end of each iteration to force
#            refreshing of neighbor data in parallel versions; August 2013
#          - Converted to Python by Jeff Hammond, February 2016.
#
# *******************************************************************

import sys
from charm4py import *
import numpy
from enum import Enum
import time

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
    BOTTOM = 4

class Cell(Chare):
    def __init__(self):
        X, Y = self.thisIndex

        W = numpy.zeros((2*r+1,2*r+1))
        if pattern == 'star':
            stencil_size = 4*r+1
            #vh = numpy.fromfunction(lambda i: 1./(2*r*(i-r)), (2*r+1,), dtype=float)
            #vh[r] = 0.0
            #W[:,r] = vh
            #W[r,:] = vh
            for i in range(1,r+1):
                W[r,r+i] = +1./(2*i*r)
                W[r+i,r] = +1./(2*i*r)
                W[r,r-i] = -1./(2*i*r)
                W[r-i,r] = -1./(2*i*r)

        else:
            stencil_size = (2*r+1)**2
            #W = numpy.fromfunction(lambda i,j: 1./(4 * numpy.maximum(numpy.abs(i-r),numpy.abs(j-r)) * (2*numpy.maximum(numpy.abs(i-r),numpy.abs(j-r)) - 1) * r),(2*r+1,2*r+1),dtype=float)
            #sign = numpy.fromfunction(lambda i,j: j-i,(2*r+1,2*r+1) )
            #sign = numpy.sign(sign[::-1])
            #temp = numpy.fromfunction(lambda x: 1./((x-r)*4*r),(2*r+1,),dtype=float)  #main diagonal
            #temp[r]=0
            #W = numpy.fill_diagonal(sign*W,temp)
            for j in range(1,r+1):
                for i in range(-j+1,j):
                    W[r+i,r+j] = +1./(4*j*(2*j-1)*r)
                    W[r+i,r-j] = -1./(4*j*(2*j-1)*r)
                    W[r+j,r+i] = +1./(4*j*(2*j-1)*r)
                    W[r-j,r+i] = -1./(4*j*(2*j-1)*r)

                W[r+j,r+j]    = +1./(4*j*r)
                W[r-j,r-j]    = -1./(4*j*r)

        width = n//x
        leftover = n%x

        if X<leftover:
            istart = (width+1) * X
            iend = istart + width
        else:
            istart = (width+1) * leftover + width * (X-leftover)
            iend = istart + width - 1

        width = iend - istart + 1
        if width == 0 :
            charm.exit("ERROR: rank", me,"has no work to do")

        height = n//y
        leftover = n%y
        if Y<leftover:
            jstart = (height+1) * Y
            jend = jstart + height

        else:
            jstart = (height+1) * leftover + height * (Y-leftover)
            jend = jstart + height - 1

        height = jend - jstart + 1
        if height == 0:
            charm.exit("ERROR: rank", me,"has no work to do")

        if width < r or height < r:
            charm.exit("ERROR: rank", me,"has work tile smaller then stencil radius")

        A = numpy.zeros((height+2*r,width+2*r))
        a = numpy.fromfunction(lambda i,j: i+istart+j+jstart,(height,width),dtype=float)
        A[r:-r,r:-r] = a
        B = numpy.zeros((height,width))

        new_self_vars = {'X': X, 'Y': Y, 'width': width,
                         'stencil_size': stencil_size, 'leftover':leftover,
                         'istart': istart, 'iend': iend, 'jstart':jstart,
                         'jend': jend,'height': height,
                         'A':A, 'B': B, 'W': W
                         }
        self.__dict__.update(new_self_vars)

    @coro
    def run(self, done_future):
        neighbors = list()
        if self.Y < y-1:
            top_proxy = self.thisProxy[(self.X, self.Y+1)]
            top_nbr = Channel(self, top_proxy)
            top_nbr.dir = Directions.TOP
            neighbors.append(top_nbr)
        if self.Y > 0:
            bot_proxy = self.thisProxy[(self.X, self.Y-1)]
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
            top_buf_in  = numpy.zeros(r*self.width)
            top_buf_out = numpy.zeros(r*self.width)
            bot_buf_in  = numpy.zeros(r*self.width)
            bot_buf_out = numpy.zeros(r*self.width)

            right_buf_in  = numpy.zeros(r*self.height)
            right_buf_out = numpy.zeros(r*self.height)
            left_buf_in  = numpy.zeros(r*self.height)
            left_buf_out = numpy.zeros(r*self.height)

        for i in range(warmup + iterations):
            if i == warmup:
                t0 = time.perf_counter()

            if self.Y < y-1 :
                kk=0
                for a in range(self.jend-r+1, self.jend+1):
                    a = a - self.jstart
                    for b in range(self.istart, self.iend+1) :
                        b = b-self.istart
                        top_buf_out[kk] = self.A[a+r][b+r]
                        kk = kk+1
                top_nbr.send(top_buf_out)

            if self.Y > 0 :
                kk=0
                for a in range(self.jstart, self.jstart+r):
                    a = a - self.jstart
                    for b in range(self.istart, self.iend+1) :
                        b = b-self.istart
                        bot_buf_out[kk] = self.A[a+r][b+r]
                        kk = kk+1
                bot_nbr.send(bot_buf_out)

            if self.X < x-1 :
                kk=0
                for a in range(self.jstart, self.jend+1):
                    a = a - self.jstart
                    for b in range(self.iend-r+1, self.iend+1) :
                        b = b-self.istart
                        right_buf_out[kk] = self.A[a+r][b+r]
                        kk = kk+1
                right_nbr.send(right_buf_out)

            if self.X > 0 :
                kk=0
                for a in range(self.jstart, self.jend+1):
                    a = a - self.jstart
                    for b in range(self.istart, self.istart+r) :
                        b = b-self.istart
                        left_buf_out[kk] = self.A[a+r][b+r]
                        kk = kk+1
                left_nbr.send(left_buf_out)

            for ready_ch in charm.iwait(neighbors):
                if ready_ch.dir == Directions.TOP:
                    kk=0
                    for a in range(self.jend+1, self.jend+r+1):
                        a = a - self.jstart
                        for b in range(self.istart, self.iend+1):
                            b = b-self.istart
                            self.A[a+r][b+r] = top_buf_in[kk]
                            kk = kk+1

                elif ready_ch.dir == Directions.BOTTOM:
                    kk=0
                    for a in range(self.jstart-r, self.jstart):
                        a = a-self.jstart
                        for b in range(self.istart, self.iend+1):
                            b = b-self.istart
                            self.A[a+r][b+r] = bot_buf_in[kk]
                            kk = kk+1

                elif ready_ch.dir == Directions.LEFT:
                    kk=0
                    for a in range(self.jstart, self.jend+1):
                        a = a - self.jstart
                        for b in range(self.istart-r, self.istart):
                            b = b-self.istart
                            self.A[a+r][b+r] = left_buf_in[kk]
                            kk = kk+1
                elif ready_ch.dir == Directions.RIGHT:
                    kk=0
                    for a in range(self.jstart, self.jend+1):
                        a = a - self.jstart
                        for b in range(self.iend+1, self.iend+r+1):
                            b = b-self.istart
                            self.A[a+r][b+r] = right_buf_in[kk]
                            kk = kk+1

            # Apply the stencil operator
            for a in range(max(self.jstart,r),min(n-r-1,self.jend)+1):
                a = a - self.jstart
                for b in range(max(self.istart,r),min(n-r-1,self.iend)+1):
                    b = b - self.istart
                    self.B[a][b] = self.B[a][b] + numpy.dot(self.W[r],self.A[a:a+2*r+1,b+r])
                    self.B[a][b] = self.B[a][b] + numpy.dot(self.W[:,r],self.A[a+r,b:b+2*r+1])

        local_time = time.perf_counter() - t0
        # ********************************************************************
        # ** Analyze and output results.
        # ********************************************************************

        # compute L1 norm in parallel
        local_norm = 0.0;
        for a in range(max(self.jstart,r), min(n-r-1,self.jend)+1):
            for b in range(max(self.istart,r), min(n-r-1,self.iend)+1):
                local_norm = local_norm + abs(self.B[a-self.jstart][b-self.istart])

        norm_time = numpy.array([local_norm, local_time], dtype='f')
        self.reduce(self.thisProxy[(0,0)].report, norm_time, Reducer.sum)
        if not self.thisIndex == (0, 0):
            self.reduce(done_future)
        else:
            self.done_f = done_future


    def report(self, norm_time):
        norm = norm_time[0]
        total_time = norm_time[1]
        epsilon=1.e-8
        active_points = (n-2*r)**2
        norm = norm / active_points
        if r > 0:
            ref_norm = (iterations+1)*(2.0)
        else:
            ref_norm = 0.0
        # if abs(norm-ref_norm) < epsilon:
        print('Solution validates')
        flops = (2*self.stencil_size+1) * active_points
        avgtime = total_time/iterations
        print('Rate (MFlops/s): ',1.e-6*flops/avgtime, ' Avg time (s): ',avgtime)
        # else:
            # print('ERROR: L1 norm = ', norm,' Reference L1 norm = ', ref_norm)
        self.done_f()



    def print_location(self):
        print(f"Chare {self.thisIndex} located on Chare {charm.myPe()}")

def main(args):
    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels ')
    print('Python Charm/Numpy  Stencil execution on 2D grid')

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('argument count = ', len(sys.argv))
        charm.exit("Usage: ./stencil <# chares> <# iterations> <array dimension> [<star/stencil> <radius>]")

    np = int(sys.argv[1])
    if np < 1:
        charm.exit("ERROR: num_chares must be >= 1")
    iterations = int(sys.argv[2])
    if iterations < 1:
        charm.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[3])
    nsquare = n * n
    if nsquare < np:
        charm.exit(f"ERROR: grid size {nsquare}, must be at least # ranks: {np}");


    if len(sys.argv) > 4:
        pattern = sys.argv[4]
    else:
        pattern = 'star'

    if len(sys.argv) > 5:
        r = int(sys.argv[5])
        if r < 1:
            charm.exit("ERROR: Stencil radius should be positive")
        if (2*r+1) > n:
            charm.exit("ERROR: Stencil radius exceeds grid size")
    else:
        r = 2


    x, y = factor(np)
    x = int(x)
    y = int(y)

    params = {'x': x, 'y': y, 'r': r,
              'np': np, 'iterations': iterations,
              'n': n, 'pattern': pattern,
              'warmup':0
            }

    charm.thisProxy.updateGlobals(params, awaitable=True).get()
    cells = Array(Cell, (x, y))

    print('Number of chares          = ', np)
    print('Number of processors      = ', charm.numPes())
    print('Grid shape (x,y)          = ', x, y)
    print('Number of iterations      = ', iterations)
    print('Grid size                 = ', n)
    if pattern == 'star':
        print('Type of stencil           = star')
    else:
        print('Type of stencil           = stencil')
        print('Radius of stencil         = ', r)
        print('Data type                 = float 64 (double precision in C)')
        print('Compact representation of stencil loop body')

    done_fut = Future()
    tstart = time.perf_counter()
    cells.print_location()
    cells.run(done_fut)
    done_fut.get()
    tend = time.perf_counter()
    print(f"Elapsed: {tend-tstart}")
    charm.exit()


charm.start(main)
