from mpi4py import MPI
import time
import numpy as np
import array
import pandas as pd
import sys

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
    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        iter_params.append((iter, msg_size))
        msg_size *= 2
    datarange = (min_msg_size, max_msg_size)
    for iter, msg_size in iter_params:
        comm.barrier()
        do_iteration(comm, rank, msg_size, windows, iter, iter_datafile_base, datarange)

def do_iteration(comm, rank, message_size, windows, num_iters, iter_datafile_base, datarange):
    local_data = np.ones(message_size, dtype='int8')
    remote_data = np.ones(message_size, dtype='int8')
    ack = np.zeros(1, dtype='int8')
    t_data = np.zeros(num_iters+WARMUP_ITERS, dtype='float64')

    partner_idx = not rank
    am_low = rank == 0

    tstart = 0

    for idx in range(num_iters + WARMUP_ITERS):
        requests = [MPI.REQUEST_NULL for _ in range(windows)]
        if idx == WARMUP_ITERS:
            tstart = time.time()
        tst = time.perf_counter()
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
        tend = time.perf_counter()
        t_data[idx] = tend-tst
    elapsed_time = time.time() - tstart
    if am_low:
        global output_df
        display_iteration_data(rank, elapsed_time, num_iters, windows, message_size)
        # iter_filename = iter_datafile_base + str(rank) + '.csv'
        # iter_data = write_iteration_data(rank, num_iters, windows, message_size, t_data)
        # if output_df is None:
        #     output_df = iter_data
        # else:
        #     output_df = pd.concat([output_df, iter_data])
        # if message_size == datarange[1]:
        #     print("Writing data now!")
        #     from datetime import datetime
        #     now = datetime.now()
        #     dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        #     iter_filename = f"{dt_string}_{iter_filename}"

        #     with open(iter_filename, 'w') as of:
        #         of.write('# ' + ' '.join(sys.argv) + '\n')
        #         output_df.to_csv(of, index=False)

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
