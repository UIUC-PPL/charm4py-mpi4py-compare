from charm4py import charm, Chare, Array, coro, Future, Channel, Group, ArrayMap
import time
import numpy as np
import array
import pandas as pd
import sys


LOW_ITER_THRESHOLD = 8192
WARMUP_ITERS = 10


class Block(Chare):
    def __init__(self, min_data, max_data):
        self.num_chares = charm.numPes()
        self.am_low_chare = self.thisIndex[0] == 0
        self.datarange = (min_data, max_data)
        self.output_df = None

        if self.am_low_chare:
            print("Chare,Msg Size, Iterations, Bandwidth (MB/s)")

    @coro
    def do_iteration(self, message_size, windows, num_iters, done_future, iter_datafile_base):
        local_data = np.ones(message_size, dtype='int8')
        remote_data = np.ones(message_size, dtype='int8')
        t_data = np.zeros(num_iters+WARMUP_ITERS, dtype='float64')

        partner_idx = int(not self.thisIndex[0])
        partner = self.thisProxy[partner_idx]
        partner_channel = Channel(self, partner)
        partner_ack_channel = Channel(self, partner)

        tstart = 0

        for idx in range(num_iters + WARMUP_ITERS):
            if idx == WARMUP_ITERS:
                tstart = time.time()
            tst = time.perf_counter()
            if self.am_low_chare:
                for _ in range(windows):
                    partner_channel.send(local_data)
                partner_ack_channel.recv()
            else:
                for _ in range(windows):
                    # The lifetime of this object has big impact on performance
                    d=partner_channel.recv()
                partner_ack_channel.send(1)
            tend = time.perf_counter()
            t_data[idx] = tend-tst
        elapsed_time = time.time() - tstart
        if self.am_low_chare:
            self.display_iteration_data(elapsed_time, num_iters, windows, message_size)
            iter_filename = iter_datafile_base + str(self.thisIndex[0]) + '.csv'
            iter_data = self.write_iteration_data(num_iters, windows, message_size, t_data)
            if self.output_df is None:
                self.output_df = iter_data
            else:
                self.output_df = pd.concat([self.output_df, iter_data])
            if message_size == self.datarange[1]:
                from datetime import datetime
                now = datetime.now()
                dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
                iter_filename = f"{dt_string}_{iter_filename}"

                with open(iter_filename, 'w') as of:
                    of.write('# ' + ' '.join(sys.argv) + '\n')
                    self.output_df.to_csv(of, index=False)

        self.reduce(done_future)

    def write_iteration_data(self, num_iters, windows, message_size, timing_data):
        header = ("Chare,Msg Size, Iteration, Bandwidth (MB/s)")
        output = pd.DataFrame(columns=header.split(','))

        timing_nowarmup = timing_data[WARMUP_ITERS::]
        per_iter_data_sent = windows * message_size / 1e6

        for elapsed_s, iteration in zip(timing_nowarmup,
                                        range(num_iters)
                                        ):
            bandwidth = (per_iter_data_sent) / elapsed_s
            iter_num = iteration + 1

            iter_data = (self.thisIndex[0], message_size, iter_num,
                         bandwidth
                         )
            output.loc[iteration] = iter_data
        return output


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
    blocks = Array(Block, 2, args=[min_msg_size, max_msg_size], map = peMap)
    charm.awaitCreation(blocks)
    msg_size = min_msg_size

    while msg_size <= max_msg_size:
        if msg_size <= LOW_ITER_THRESHOLD:
            iter = low_iter
        else:
            iter = high_iter
        done_future = Future()
        blocks.do_iteration(msg_size, window_size, iter, done_future, iter_datafile_base)
        done_future.get()
        msg_size *= 2

    charm.exit()


charm.start(main)
