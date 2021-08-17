from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
import array
import pandas as pd
import sys
import random
import gc


LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10


class Block(Chare):
    def __init__(self):
        self.num_chares = charm.numPes()
        self.am_low_chare = self.thisIndex[0] == 0
        self.output_df = None
        partner_idx = int(not self.thisIndex[0])
        self.partner = self.thisProxy[partner_idx]
        self.partner_channel = Channel(self, self.partner)
        self.partner_ack_channel = Channel(self, self.partner)
        self.windows = 0

        if self.am_low_chare:
            print("Chare,Msg Size, Iterations, Bandwidth (MB/s)")

    @coro
    def do_iteration(self, message_size, windows, num_iters, done_future, iter_datafile_base):
        self.windows = windows
        local_data = np.ones(message_size, dtype='int8')
        remote_data = np.ones(message_size, dtype='int8')
        partner_channel = self.partner_channel
        partner_ack_channel = self.partner_ack_channel

        tstart = 0

        for idx in range(num_iters + WARMUP_ITERS):
            iter_st = time.perf_counter()
            if idx == WARMUP_ITERS:
                tstart = time.perf_counter()
            if self.am_low_chare:
                for _ in range(windows):
                    partner_channel.send(local_data)
                partner_ack_channel.recv()
            else:
                for _ in range(windows):
                    # The lifetime of this object has big impact on performance
                    d=partner_channel.recv()
                partner_ack_channel.send(1)
            iter_e = time.perf_counter()
            if self.am_low_chare:
                self.iteration_data[self.completed_iterations] = (message_size,
                                                                  idx,
                                                                  num_iters,
                                                                  WARMUP_ITERS,
                                                                  iter_e - iter_st
                                                                  )
                self.completed_iterations += 1
        elapsed_time = time.perf_counter() - tstart
        if self.am_low_chare:
            self.display_iteration_data(elapsed_time, num_iters, windows, message_size)
            if self.completed_iterations == self.total_iterations:
                self.write_output(iter_datafile_base)
        self.reduce(done_future)

    def receive_params(self, iter_params):
        msg_iters = [x[0] for x in iter_params]
        self.total_iterations = sum(msg_iters)
        self.total_iterations += len(iter_params) * WARMUP_ITERS
        # Message size, iteration, total iterations, warmup iterations, bandwidth (or time)
        self.iteration_data = np.ndarray((self.total_iterations, 5), dtype=np.float64)
        self.completed_iterations = 0

    def write_output(self, filename_base):
        import pandas as pd
        header = ("Message size", "Iteration", "Total Iterations",
                  "Warmup", "Bandwidth (MB/s)"
                  )
        df = pd.DataFrame(self.iteration_data, columns=header)
        time = df['Bandwidth (MB/s)']
        data_volume = df['Message size'] * self.windows / 1e6
        bw = data_volume / time
        df['Bandwidth (MB/s)'] = bw

        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = f"{dt_string}_{filename_base}.csv"
        with open(output_filename, 'w') as of:
            of.write('# ' + ' '.join(sys.argv) + '\n')
            df.to_csv(of, index=False)


    def display_iteration_data(self, elapsed_time, num_iters, windows, message_size):
        data_sent = message_size / 1e6 * num_iters * windows
        print(f'{self.thisIndex[0]},{message_size},{num_iters},{data_sent/elapsed_time}')



class ArrMap(ArrayMap):
    def procNum(self, index):
        return index[0] % 2


def main(args):
    if len(args) < 6:
        print("Doesn't have the required input params. Usage:"
              "<min-msg-size> <max-msg-size> <window-size> "
              "<low-iter> <high-iter>\n"
              )
        charm.exit(-1)

    min_msg_size = int(args[1])
    max_msg_size = int(args[2])
    window_size = int(args[3])
    low_iter = int(args[4])
    high_iter = int(args[5])
    if len(args) == 7:
        iter_datafile_base = args[6]
    else:
        iter_datafile_base = None

    peMap = Group(ArrMap)
    blocks = Array(Block, 2, args=[], map=peMap)
    charm.awaitCreation(blocks)
    msg_size = min_msg_size
    iter_params = []

    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        iter_params.append((iter, msg_size))
        msg_size *= 2

    blocks[0].receive_params(iter_params, awaitable=True).get()
    for iter, msg_size in iter_params:
        done_future = Future()
        blocks.do_iteration(msg_size, window_size, iter, done_future, iter_datafile_base)
        done_future.get()
    charm.exit()


charm.start(main)
