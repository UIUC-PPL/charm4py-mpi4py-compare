from charm4py import charm, Chare, Array, coro, Future, Channel, Group
import time
import numpy as np
import sys
warmup = 60

class Ping(Chare):
    def __init__(self, print_format):
        self.num_chares = charm.numPes()
        self.print_format = print_format
        self.am_low_chare = self.thisIndex == 0

        if self.am_low_chare:
            self.iter_timing = list()
            if print_format == 0:
                print("Msg Size, Iterations, One-way Time (us), "
                      "Bandwidth (bytes/us)"
                      )
            else:
                print(f'{"Msg Size": <30} {"Iterations": <25} '
                      f'{"One-way Time (us)": <20} {"Bandwidth (bytes/us)": <20}'
                      )

    @coro
    def do_iteration(self, message_size, num_iters, done_future):
        data = np.zeros(message_size, dtype='int8')
        partner_idx = int(not self.thisIndex)
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)

        tstart = time.perf_counter_ns()

        for i in range(num_iters + warmup):
            if i == warmup:
                tstart = time.perf_counter_ns()
            if self.am_low_chare:
                partner_channel.send(data)
                # If we don't capture this,
                # we are timing extra deallocation costs
                # for large messages, it makes a
                # massive difference
                d=partner_channel.recv()

            else:
                d=partner_channel.recv()
                partner_channel.send(data)
        tend = time.perf_counter_ns()

        elapsed_time = tend - tstart

        if self.am_low_chare:
            self.display_iteration_data(elapsed_time, num_iters, message_size)
        self.reduce(done_future)

    def display_iteration_data(self, elapsed_time, num_iters, message_size):
        elapsed_time /= 2  # 1-way performance, not RTT
        elapsed_time /= num_iters  # Time for each message
        elapsed_time /= 1e9
        bandwidth = message_size / elapsed_time
        if self.print_format == 0:
            print(f'{message_size},{num_iters},{elapsed_time * 1e6},'
                  f'{bandwidth / 1e6}'
                  )
        else:
            print(f'{message_size: <30} {num_iters: <25} '
                  f'{elapsed_time * 1e6: <20} {bandwidth / 1e6: <20}'
                  )

    def receive_params(self, iter_params):
        msg_iters = [x[0] for x in iter_params]
        self.total_iterations = sum(msg_iters)
        self.total_iterations += len(iter_params) * warmup
        # Message size, group, total iterations, warmup iterations, latency (or time)
        self.iteration_data = np.ndarray((self.total_iterations//groupsize, 5),
                                         dtype=np.float64
                                         )
        self.completed_iterations = 0

    def write_output(self, filename):
        import pandas as pd
        header = ("Message size", "Grouping", "Total Iterations",
                  "Warmup", "Latency (us)"
                  )
        df = pd.DataFrame(self.iteration_data, columns=header)
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

def main(args):
    if len(args) < 6:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <low-iter> "
              "<high-iter> <print-format"
              "(0 for csv, 1 for "
              "regular)> <output_file>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    low_iter = int(args[3])
    high_iter = int(args[4])
    print_format = int(args[5])
    try:
        output_file = args[6]
    except:
        output_file = ''

    pings = Group(Ping, args=[print_format])
    charm.awaitCreation(pings)
    msg_size = min_msg_size
    iter_order = list()

    while msg_size <= max_msg_size:
        if msg_size <= 1048576:
            iter = low_iter
        else:
            iter = high_iter
        iter_order.append((iter, msg_size))
        msg_size *= 2

    pings[0].receive_params(iter_order)
    for iter, msg_size in iter_order:
        done_future = Future()
        pings.do_iteration(msg_size, iter, done_future)
        done_future.get()
    charm.exit()


charm.start(main)

