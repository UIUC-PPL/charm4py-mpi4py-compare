#!/usr/bin/env python3
import mpi4py.rc; mpi4py.rc.threads = False
import sys
from mpi4py import MPI
import numpy
import time
from numba import cuda

COMM_TIME, COMP_TIME, ITER_TIME = range(3)

def factor(r):
    fac1 = int(numpy.sqrt(r+1.0))
    fac2 = 0
    for fac1 in range(fac1, 0, -1):
        if r%fac1 == 0:
            fac2 = r/fac1
            break;
    return fac1, fac2

def output_timing(data, n, m, np, iterations, warmup, output_fp):
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_filename = f"{output_fp}/{dt_string}_n{n}_m{n}_np{np}_i{iterations}_w{warmup}_mpi.csv"
    header = "Process,Iteration,Iteration Time,Communication Time,Computation Time"
    with open(output_filename, 'w') as output_file:
        output_file.write(f'#{" ".join(sys.argv)}\n')
        output_file.write(header + '\n')
        for pnum, timing in enumerate(data):
            for inum, iter in enumerate(timing):
                in_seconds = iter/1e9
                comm_time = in_seconds[COMM_TIME]
                comp_time = in_seconds[COMP_TIME]
                iter_time = in_seconds[ITER_TIME]
                output_tuple = (pnum, inum+1, iter_time, comm_time, comp_time)
                output_file.write(','.join(map(str, output_tuple)) + '\n')

def main():

    comm = MPI.COMM_WORLD
    me = comm.Get_rank() #My ID
    np = comm.Get_size() #Number of processor, NOT numpy
    ngpus = len(cuda.gpus)
    cuda.select_device(me%ngpus)
    import gpu_kernels as kernels
    x, y = factor(np)
    comm = comm.Create_cart([x,y])
    coords = comm.Get_coords(me)
    X = coords[0]
    Y = coords[1]

    x = int(x)
    y = int(y)
    output_filepath = '.'
    try:
        arg_int = int(sys.argv[-1])
    except:
        output_filepath = sys.argv[-1]

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

    t_template = numpy.zeros(my_blocksize + ghost_size, dtype=numpy.float64)
    T = cuda.device_array_like(t_template)
    newT = cuda.device_array_like(t_template)

    kernels.enforce_BC(T)

    top_buf_out_h = numpy.zeros(width)
    top_buf_in_h = numpy.zeros(width)
    bot_buf_out_h = numpy.zeros(width)
    bot_buf_in_h = numpy.zeros(width)

    top_buf_out = cuda.to_device(top_buf_out_h)
    top_buf_in = cuda.to_device(top_buf_in_h)
    bot_buf_out = cuda.to_device(bot_buf_out_h)
    bot_buf_in = cuda.to_device(bot_buf_in_h)

    right_buf_out_h = numpy.zeros(height)
    right_buf_in_h = numpy.zeros(height)
    left_buf_out_h = numpy.zeros(height)
    left_buf_in_h = numpy.zeros(height)

    right_buf_out = cuda.to_device(right_buf_out_h)
    right_buf_in = cuda.to_device(right_buf_in_h)
    left_buf_out = cuda.to_device(left_buf_out_h)
    left_buf_in = cuda.to_device(left_buf_in_h)

    if Y < y-1:
        top_nbr   = comm.Get_cart_rank([X,Y+1])
    if Y > 0:
        bot_nbr   = comm.Get_cart_rank([X,Y-1])
    if X > 0:
        left_nbr  = comm.Get_cart_rank([X-1,Y])
    if X < x-1:
        right_nbr = comm.Get_cart_rank([X+1,Y])

    # information for compute, comm., and total time for all iterations
    # We consider communication time the time it takes to pack, send, and
    # unpack ghosts.
    # Computation is just time spent in the kernel and enforcing BC
    timing_info = numpy.zeros((warmup + iterations, 3), dtype=numpy.float64)

    for i in range(warmup + iterations):
        if i<1:
            comm.Barrier()
            t0 = MPI.Wtime()
        if i == warmup:
            tst = MPI.Wtime()

        iter_start = time.perf_counter_ns()
        comm_start = iter_start
        send_reqs = list()
        recv_reqs = list()
        recv_status = MPI.Status()
        if Y < y-1 :
            req0 = comm.Irecv(top_buf_in, source =top_nbr , tag =101)
            kernels.pack_top(T, top_buf_out)
            req1 = comm.Isend(top_buf_out, dest =top_nbr, tag =99)
            recv_reqs.append(req0)
            send_reqs.append(req1)

        if Y > 0 :
            req2 = comm.Irecv(bot_buf_in, source =bot_nbr , tag =99)
            kernels.pack_bottom(T, bot_buf_out)
            req3 = comm.Isend(bot_buf_out, dest =bot_nbr, tag =101)
            recv_reqs.append(req2)
            send_reqs.append(req3)

        if X < x-1 :
            req4 = comm.Irecv(right_buf_in, source =right_nbr , tag =1010)
            kernels.pack_right(T, right_buf_out)
            req5 = comm.Isend(right_buf_out, dest =right_nbr, tag =990)
            recv_reqs.append(req4)
            send_reqs.append(req5)

        if X > 0 :
            req6 = comm.Irecv(left_buf_in, source =left_nbr , tag =990)
            kernels.pack_left(T, left_buf_out)
            req7 = comm.Isend(left_buf_out, dest =left_nbr, tag =1010)
            recv_reqs.append(req6)
            send_reqs.append(req7)

        for _ in recv_reqs:
            MPI.Request.Waitany(recv_reqs, status=recv_status)
            tag = recv_status.Get_tag()
            if tag == 101:
                kernels.unpack_top(T, top_buf_in)
            elif tag == 99:
                kernels.unpack_bottom(T, bot_buf_in)
            elif tag == 990:
                kernels.unpack_left(T, left_buf_in)
            elif tag == 1010:
                kernels.unpack_right(T, right_buf_in)


        MPI.Request.Waitall(send_reqs)
        comm_end = time.perf_counter_ns()
        comp_start = comm_end

        kernels.compute(newT, T)
        newT, T = T, newT
        kernels.enforce_BC(T)

        iter_end = time.perf_counter_ns()
        comp_end = iter_end

        timing_info[i][COMM_TIME] = comm_end - comm_start
        timing_info[i][COMP_TIME] = comp_end - comp_start
        timing_info[i][ITER_TIME] = iter_end - iter_start
    tend = MPI.Wtime()
    if me == 0:
        print(f"Elapsed: {tend-tst}")

    comm = MPI.COMM_WORLD
    data = comm.gather(timing_info, root=0)
    if me == 0:
        output_timing(data, n, m, np, iterations, warmup, output_filepath)


if __name__ == '__main__':
    main()
