from mpi4py import MPI
import time
import numpy as np
import sys
warmup = 60

def main():
    args = sys.argv
    if len(args) < 6:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <low-iter> "
              "<high-iter> <print-format"
              "(0 for csv, 1 for "
              "regular)> <output_file>\n"
              )
        charm.exit(-1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    low_iter = int(args[3])
    high_iter = int(args[4])
    print_format = int(args[5])
    try:
        output_file = args[6]
    except:
        output_file = ''

    msg_size = min_msg_size
    iter_order = list()
    total_iters = 0

    while msg_size <= max_msg_size:
        if msg_size <= 1048576:
            iter = low_iter
        else:
            iter = high_iter
        total_iters += iter + warmup

        iter_order.append((iter, msg_size))
        msg_size *= 2


    for iter, msg_size in iter_order:
        do_iteration(comm, rank, msg_size, iter)
    sys.exit()

def do_iteration(comm, rank, message_size, num_iters):
    data = np.zeros(message_size, dtype='int8')
    data_recv = np.zeros(message_size, dtype='int8')
    partner = not rank
    am_low = rank == 0

    tstart = time.perf_counter_ns()
    for i in range(num_iters + warmup):
        if i == warmup:
            comm.barrier()
            iter_st = time.perf_counter_ns()
            tstart = time.perf_counter_ns()
        if rank == 0:
            comm.Send(data, dest=partner, tag=0)
            comm.Recv(data_recv, source=partner, tag=1)
        else:
            comm.Recv(data_recv, source=partner, tag=0)
            comm.Send(data, dest=partner, tag=1)

    tend = time.perf_counter_ns()
    elapsed_time = tend - tstart

    if rank == 0:
        display_iteration_data(elapsed_time, num_iters, message_size)

def display_iteration_data(elapsed_time, num_iters, message_size):
    elapsed_time /= 2  # 1-way performance, not RTT
    elapsed_time /= num_iters  # Time for each message
    elapsed_time /= 1e9
    bandwidth = message_size / elapsed_time
    print(f'{message_size},{num_iters},{elapsed_time * 1e6},'
          f'{bandwidth / 1e6}'
          )

def write_output(filename, iteration_data):
    import pandas as pd
    header = ("Message size", "Grouping", "Total Iterations",
              "Warmup", "Latency (us)"
              )
    df = pd.DataFrame(iteration_data, columns=header)
    # convert latency from round-trip
    # groupsize iterations in nanoseconds to
    # average one-way time per iteration in us
    df['Latency (us)'] /= (groupsize*1e9*2)/1e6

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_filename = f"{dt_string}_{filename}.csv"
    with open(output_filename, 'w') as of:
        of.write('# ' + ' '.join(sys.argv) + '\n')
        df.to_csv(of, index=False)


if __name__ == '__main__':
    main()
