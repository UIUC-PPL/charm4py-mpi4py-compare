from mpi4py import MPI
import time
import numpy as np
import array
import pandas as pd
import sys
import random
from numba import cuda

output_df = None

LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10

def main():
    args = sys.argv
    if len(args) < 6:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <window-size> "
              "<low-iter> <high-iter>\n"
              )
        exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    if numprocs != 2:
        if myid == 0:
            errmsg = "This test requires exactly two processes"
        else:
            errmsg = None
        raise SystemExit(errmsg)


    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    windows = int(args[3])
    low_iter = int(args[4])
    high_iter = int(args[5])
    if len(args) == 7:
        iter_datafile_base = args[6]
    else:
        iter_datafile_base = None

    iter_params = []
    msg_size = min_msg_size
    total_iterations = 0
    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        total_iterations += iter + WARMUP_ITERS
        iter_params.append((iter, msg_size))
        msg_size *= 2

    if rank == 0:
        iteration_data = np.ndarray((total_iterations, 5), dtype=np.float64)
        print("Process,Msg Size, Iterations, Bandwidth (MB/s)")
    else:
        iteration_data = None

    for iter, msg_size in iter_params:
        do_iteration(comm, rank, msg_size, windows, iter, iteration_data)
    if rank == 0:
        write_output(iter_datafile_base, iteration_data, windows)
    sys.exit(0)


def write_output(filename_base, iteration_data, windows):
    import pandas as pd
    header = ("Message size", "Iteration", "Total Iterations",
              "Warmup", "Bandwidth (MB/s)"
              )
    df = pd.DataFrame(iteration_data, columns=header)
    time = df['Bandwidth (MB/s)']
    data_volume = df['Message size'] * windows / 1e6
    bw = data_volume / time
    df['Bandwidth (MB/s)'] = bw

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_filename = f"{dt_string}_{filename_base}.csv"
    with open(output_filename, 'w') as of:
        of.write('# ' + ' '.join(sys.argv) + '\n')
        df.to_csv(of, index=False)


def do_iteration(comm, rank, message_size, windows, num_iters, iter_data=None):
    local_data = np.ones(message_size, dtype='int8')
    remote_data = np.ones(message_size, dtype='int8')
    ack = np.zeros(1, dtype='int8')

    partner_idx = not rank
    am_low = rank == 0

    tstart = 0

    for idx in range(num_iters + WARMUP_ITERS):
        iter_start = time.perf_counter()
        requests = [MPI.REQUEST_NULL for _ in range(windows)]
        if idx == WARMUP_ITERS:
            comm.barrier()
            iter_start = time.perf_counter()
            tstart = time.perf_counter()
        if am_low:
            for window in range(windows):
                requests[window] = comm.Isend(local_data, tag=0, dest=partner_idx)
            MPI.Request.Waitall(requests)
            comm.Recv(ack, tag=1, source=partner_idx)
        else:
            for window in range(windows):
                requests[window] = comm.Irecv(remote_data, tag=0, source=partner_idx)
            MPI.Request.Waitall(requests)
            comm.Send(ack, tag=1, dest=partner_idx)
        iter_end = time.perf_counter()
        if iter_data is not None:
            iter_data[do_iteration.iter_count] = (message_size,
                                                  idx,
                                                  num_iters,
                                                  WARMUP_ITERS,
                                                  iter_end - iter_start
                                                  )
            do_iteration.iter_count += 1
    elapsed_time = time.perf_counter() - tstart
    if am_low:
        display_iteration_data(rank, elapsed_time, num_iters, windows, message_size)
    comm.barrier()


do_iteration.iter_count = 0


def write_iteration_data(rank, num_iters, windows, message_size, timing_data):
   header = ("Process,Msg Size, Iteration, Bandwidth (MB/s)")
   output = pd.DataFrame(columns=header.split(','))

   timing_nowarmup = timing_data[WARMUP_ITERS::]
   per_iter_data_sent = windows * message_size / 1e6

   for elapsed_s, iteration in zip(timing_nowarmup,
                                   range(num_iters)
                                   ):
       bandwidth = (per_iter_data_sent) / elapsed_s
       iter_num = iteration + 1

       iter_data = (rank, message_size, iter_num,
                    bandwidth
                    )
       output.loc[iteration] = iter_data
   return output


def display_iteration_data(rank,elapsed_time, num_iters, windows, message_size):
    data_sent = message_size / 1e6 * num_iters * windows
    print(f'{rank},{message_size},{num_iters},{data_sent/elapsed_time}')

if __name__ == '__main__':
    main()
