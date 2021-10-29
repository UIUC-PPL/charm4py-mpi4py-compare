from __future__ import absolute_import, division, print_function
from charm4py import charm, Chare, Array, Future, Reducer
import sys
import time
# import task_bench_core as core
import cffi
import os
import subprocess
import numpy as np

SENDING = False
RECEIVING = True

root_dir = os.path.dirname(os.path.dirname(__file__))
core_header = subprocess.check_output(
    [
        "gcc", "-D", "__attribute__(x)=", "-E", "-P",
        os.path.join(root_dir, "core/core_c.h")
    ]).decode("utf-8")
ffi = cffi.FFI()
ffi.cdef(core_header)
c = ffi.dlopen("libcore.so")

def app_create(args):
    c_args = []
    c_argv = ffi.new("char *[]", len(args) + 1)
    for i, arg in enumerate(args):
        c_args.append(ffi.new("char []", arg.encode('utf-8')))
        c_argv[i] = c_args[-1]
    c_argv[len(args)] = ffi.NULL

    app = c.app_create(len(args), c_argv)
    # c.app_display(app)
    return app

def init_scratch_direct(scratch_bytes):
    scratch = np.empty(scratch_bytes, dtype=np.ubyte)
    scratch_ptr = ffi.cast("char *", scratch.ctypes.data)
    c.task_graph_prepare_scratch(scratch_ptr, scratch_bytes)
    return scratch

def execute_point_impl(graph, timestep, point, scratch, *inputs):
    # if (len(inputs) > 0):
    #     #print("inputs[0] = {};  data = {}".format(inputs[0], inputs[0].ctypes.data))

    input_ptrs = ffi.new(
        "char *[]", [ffi.cast("char *", i.ctypes.data) for i in inputs])
    input_sizes = ffi.new("size_t []", [i.shape[0] for i in inputs])

    output = np.empty(graph.output_bytes_per_task, dtype=np.ubyte)
    output_ptr = ffi.cast("char *", output.ctypes.data)

    if scratch is not None:
        scratch_ptr = ffi.cast("char *", scratch.ctypes.data)
        scratch_size = scratch.shape[0]
    else:
        scratch_ptr = ffi.NULL
        scratch_size = 0

    c.task_graph_execute_point_scratch(
        graph, timestep, point, output_ptr, output.shape[0], input_ptrs,
        input_sizes, len(inputs), scratch_ptr, scratch_size)

    return output

class SubChare(Chare):
    def __init__(self, main_proxy):
        self.main_proxy = main_proxy
        #print(self.main_proxy)

    def runTimestep(self):

        offset = c.task_graph_offset_at_timestep(self.graph, self.current_timestep)
        width = c.task_graph_width_at_timestep(self.graph, self.current_timestep)
        thisIndex = self.thisIndex[0]
        if (offset <= thisIndex and thisIndex < offset + width):
            out = execute_point_impl(self.graph, self.current_timestep, thisIndex, self.scratch, *self.inputs[self.current_timestep])
            self.output = out
        #print("SC[{}] @ {}  out = {}".format(thisIndex, self.current_timestep, self.output))

        for target in self.where_to_send[self.current_timestep]:
            # #print("target = {}".format(target))
            self.thisProxy[target].receive(self.output)
            # #print("")
        self.sent = True
        self.checkAndRun(SENDING)

    def receive(self, input):
        point = np.frombuffer(input.tobytes(), dtype=np.uint64)
        #print("SC[{}] @ {}  get input = {}; type = {}; dtype = {}; point = {}".format(self.thisIndex, self.current_timestep, input, type(input), input.dtype, point))

        point_0 = int(point[0]+1)
        point_1 = int(point[1])
        self.not_received[point_0].remove(point_1)
        idx = self.receiving_map[(point_0, point_1)]
        self.inputs[point_0][idx] = input

        self.checkAndRun(RECEIVING)

    def checkAndRun(self, receiving):
        if len(self.not_received[self.current_timestep+1]) == 0:
            if self.current_timestep + 1 == self.graph.timesteps -1:
                self.reduce(self.main_proxy[0].finishedGraph, 1, Reducer.sum)
            else:
                self.sent = False
                self.current_timestep += 1
                self.runTimestep()

    def initGraph(self, gidx=None, args=None):
        # #print("gidx = {}; args = {}".format(gidx, args))
        if gidx is not None:
            self.app = app_create(args)
            self.graph = c.task_graph_list_task_graph(c.app_task_graphs(self.app), gidx)
        self.current_timestep = 0
        self.sent = False
        self.inputs = [None] * self.graph.timesteps
        self.not_received = [None] * self.graph.timesteps
        self.where_to_send = [None] * self.graph.timesteps
        self.output = np.zeros(self.graph.output_bytes_per_task, dtype=np.ubyte)
        self.scratch = init_scratch_direct(self.graph.scratch_bytes_per_task)
        self.receiving_map = dict()
        #print("gidx = {}; output_bytes_per_task = {}, scratch_bytes_per_task = {}".format(gidx, self.graph.output_bytes_per_task, self.graph.scratch_bytes_per_task))

        thisIndex = self.thisIndex[0]
        #print(thisIndex)
        for t in range(self.graph.timesteps):
            self.inputs[t] = []
            offset = c.task_graph_offset_at_timestep(self.graph, t)
            width = c.task_graph_width_at_timestep(self.graph, t)
            last_offset = 0
            last_width = 0
            if (t > 0):
                last_offset = c.task_graph_offset_at_timestep(self.graph, t-1)
                last_width = c.task_graph_width_at_timestep(self.graph, t-1)
            next_offset = c.task_graph_offset_at_timestep(self.graph, t+1)
            next_width = c.task_graph_width_at_timestep(self.graph, t+1)


            #print("SC[{}] @ {} offset {}; width {}; last_offset {}; last_width {}; next_offset {}; next_width {}".format(thisIndex, t, offset, width, last_offset, last_width, next_offset, next_width))
            dset = c.task_graph_dependence_set_at_timestep(self.graph, t)
            deps = c.task_graph_dependencies(self.graph, dset, thisIndex)
            #print("SC[{}] @ {} dset = {}; deps = {}; c.interval_list_num_intervals(deps) = {}".format(thisIndex, t, dset, deps, c.interval_list_num_intervals(deps)))

            # dependencies
            if (t == 0) or (thisIndex < offset) or (thisIndex >= width + offset):
                self.not_received[t] = set()
            else :     
                deps_t = []
                for i in range(0, c.interval_list_num_intervals(deps)):
                    interval = c.interval_list_interval(deps, i)
                    for dep in range(interval.start, interval.end + 1):
                        if (dep >= last_offset and dep < last_width + last_offset):
                            # #print("dep = {}".format(deps))
                            deps_t.append(dep)
                self.not_received[t] = set(deps_t)

                #print("SC[{}] @ {} not_received = {}".format(thisIndex, t, self.not_received))

            tset = c.task_graph_dependence_set_at_timestep(self.graph, t+1)
            tars = c.task_graph_reverse_dependencies(self.graph, tset, thisIndex)
            #print("SC[{}] @ {} tset = {}; tars = {}; c.interval_list_num_intervals(tars) = {}".format(thisIndex, t, tset, tars, c.interval_list_num_intervals(tars)))
            
            # reverse dependencies
            if (t == self.graph.timesteps - 1) or (thisIndex < offset) or (thisIndex >= width + offset):
                self.where_to_send[t] = set()
            else:
                tars_t = []
                for i in range(0, c.interval_list_num_intervals(tars)):
                    interval = c.interval_list_interval(tars, i)
                    for tar in range(interval.start, interval.end + 1):
                        if (tar >= next_offset and tar < next_width + next_offset):
                            # #print("dep = {}".format(deps))
                            tars_t.append(tar)
                self.where_to_send[t] = set(tars_t)
                #print("SC[{}] @ {} where_to_send = {}".format(thisIndex, t, self.where_to_send))


            # generate tasks
            tidx = 0
            for i in range(0, c.interval_list_num_intervals(deps)):
                interval = c.interval_list_interval(deps, i)
                for dep in range(interval.start, interval.end + 1):
                    if (dep >= last_offset and dep < last_width + last_offset):
                        #print("SC[{}] @ {} tidx = {}".format(thisIndex, t, tidx))
                        self.receiving_map[(t, dep)] = tidx
                        input_bytes_per_task = self.graph.output_bytes_per_task
                        self.inputs[t].append(np.zeros(input_bytes_per_task, dtype=np.ubyte))
                        # self.input_ptrs[t].append(const_cast<char *>(inputs[timestep].back().data()));
                        tidx += 1
        #print("SC[{}] @ {} inputs[t] = {}; input_bytes[t] = {};\nreceiving_map = {}".format(qthisIndex, t, self.inputs, self.input_bytes, self.receiving_map))
        self.reduce(self.main_proxy[0].workerReady, 1, Reducer.sum)

    def reset(self):
        self.initGraph()

class Main(Chare):
    def __init__(self, *args):
        self.args = args
        self.num_runs = 1
        self.num_runs_done = 0
        self.total_time_elapsed = .0
        #print("args = {}".format(args))

    def init_tasks(self):
        app = app_create(self.args)
        self.app = app
        c.app_display(app)
        #print("thisProxy = {}".format(self.thisProxy))
        arr_ids = []
        graphs = c.app_task_graphs(app)
        for i in range(c.task_graph_list_num_task_graphs(graphs)):
            graph = c.task_graph_list_task_graph(graphs, i)
            #print("i = {}; task_graph_list_task_graph(graphs, i) = {}".format(i, c.task_graph_list_task_graph(graphs, i)))
            arr_size = graph.max_width
            #print("arr_size = {}".format(arr_size))
            subchare_proxy = Array(SubChare, arr_size, args=[self.thisProxy])
            arr_ids.append(subchare_proxy)

        #print("arr_ids = {}".format(arr_ids))
        section_proxy = arr_ids[0]
        self.section_proxy = section_proxy
        #print("section_proxy = {}".format(section_proxy))
        section_proxy.initGraph(0, self.args)

    def workerReady(self, res):
        #print("In workerReady, res = {}".format(res))
        self.start_time = time.perf_counter()
        self.section_proxy.runTimestep()

    def finishedGraph(self, res):
        total_time = time.perf_counter() - self.start_time
        self.num_runs_done +=1
        #print("Time for last run: {}".format(total_time))
        if self.num_runs_done > 1:
            self.total_time_elapsed += total_time
        if self.num_runs_done == self.num_runs + 1:
            c.app_report_timing(self.app, total_time)
            exit()
        else:
            self.section_proxy.reset()


def main(args):
    # app = app_create(args)
    main_proxy = Chare(Main, args)
    main_proxy.init_tasks()
    # exit()

charm.start(main)