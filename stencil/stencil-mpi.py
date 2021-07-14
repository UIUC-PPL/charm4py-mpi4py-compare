#!/usr/bin/env python3
import sys
from mpi4py import MPI
import numpy
import kernels

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = r/fac1
            break;
    return fac1, fac2

def main():

    comm = MPI.COMM_WORLD
    me = comm.Get_rank() #My ID
    np = comm.Get_size() #Number of processor, NOT numpy
    x, y = factor(np)
    comm = comm.Create_cart([x,y])
    coords = comm.Get_coords(me)
    X = coords[0]
    Y = coords[1]

    x = int(x)
    y = int(y)

    if me == 0:
        print(f"X, y: {x} {y}")

    if me==0:
        print('Python MPI/Numpy  Stencil execution on 2D grid')

        if len(sys.argv) < 3:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: ./stencil <# iterations> [<array dimension> or <array dimension X> <array dimension Y>]")
    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    n = int(sys.argv[2])
    if len(sys.argv) > 3:
        m = int(sys.argv[3])
    else:
        m = n

    nsquare = n * m

    if nsquare < np:
        sys.exit("ERROR: grid size ", nsquare, " must be at least # ranks: ", Num_procs);
    if n % x:
        sys.exit(f"ERROR: grid size {n} does not evenly divide the number of chares in the x dimension {x}")
    if m % y:
        sys.exit(f"ERROR: grid size {m} does not evenly divide the number of chares in the y dimension {y}")

    warmup=10
    if me == 0:
        print('Number of processors        = ', np)
        print('Grid shape (x,y)            = ', x, y)
        print('Problem Domain (x,y)        = ', n, m)
        print('Number of warmup iterations = ', warmup)
        print('Number of iterations        = ', iterations)

    my_blocksize = (n//x)*(m//y)
    ghost_size = ((n//x) * 2) + ((m//y) * 2)
    kernels.set_block_params(m//y, n//x)

    width = m//y
    height = n//x

    T = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)
    newT = numpy.ones(my_blocksize + ghost_size, dtype=numpy.float64)

    top_buf_out = numpy.zeros(width)
    top_buf_in = numpy.zeros(width)
    bot_buf_out = numpy.zeros(width)
    bot_buf_in = numpy.zeros(width)

    right_buf_out = numpy.zeros(height)
    right_buf_in = numpy.zeros(height)
    left_buf_out = numpy.zeros(height)
    left_buf_in = numpy.zeros(height)


    if Y < y-1:
        top_nbr   = comm.Get_cart_rank([X,Y+1])
    if Y > 0:
        bot_nbr   = comm.Get_cart_rank([X,Y-1])
    if X > 0:
        left_nbr  = comm.Get_cart_rank([X-1,Y])
    if X < x-1:
        right_nbr = comm.Get_cart_rank([X+1,Y])

    for i in range(warmup + iterations):
        if i<1:
            comm.Barrier()
            t0 = MPI.Wtime()
        if i == warmup:
            tst = MPI.Wtime()
        if Y < y-1 :
            req0 = comm.Irecv(top_buf_in, source =top_nbr , tag =101 )
            kernels.pack_top(T, top_buf_out)
            req1 = comm.Isend(top_buf_out, dest =top_nbr, tag =99)

        if Y > 0 :
            req2 = comm.Irecv(bot_buf_in, source =bot_nbr , tag =99 )
            kernels.pack_bottom(T, bot_buf_out)
            req3 = comm.Isend(bot_buf_out, dest =bot_nbr, tag =101)

        if X < x-1 :
            req4 = comm.Irecv(right_buf_in, source =right_nbr , tag =1010)
            kernels.pack_right(T, right_buf_out)
            req5 = comm.Isend(right_buf_out, dest =right_nbr, tag =990)

        if X > 0 :
            req6 = comm.Irecv(left_buf_in, source =left_nbr , tag =990 )
            kernels.pack_left(T, left_buf_out)
            req7 = comm.Isend(left_buf_out, dest =left_nbr, tag =1010)

        if Y < y-1 :
            req0.wait()
            req1.wait()
            kernels.unpack_top(T, top_buf_in)
        if Y > 0 :
            req2.wait()
            req3.wait()
            kernels.unpack_bottom(T, bot_buf_in)
        if X > 0 :
            req6.wait()
            req7.wait()
            kernels.unpack_left(T, left_buf_in)

        if X < x-1 :
            req4.wait()
            req5.wait()
            kernels.unpack_right(T, right_buf_in)

        kernels.compute(newT, T)
        newT, T = T, newT
        kernels.enforce_BC(T)
    tend = MPI.Wtime()
    if me == 0:
        print(f"Elapsed: {tend-tst}")


if __name__ == '__main__':
    main()
