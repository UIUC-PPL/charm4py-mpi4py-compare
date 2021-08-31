import sh
import sys
import random
import time
import os
import sys


def main():
    hostname = sys.argv[1]
    os.putenv('UCX_RNDV_THRESH', '131072')
    os.putenv('UCX_MEMTYPE_CACHE', 'n')
    mpirun_args = ['-H', f'{hostname}:2', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH',
                   '-x', 'UCX_RNDV_THRESH=131072', '-x', 'UCX_MEMTYPE_CACHE=n',
                   ]

    charm_args = ['+ppn', '1' '+pemap', 'L0,4']

    # UCX_RNDV_THRESH=131072 UCX_MEMTYPE_CACHE=n jsrun -n2 -a1 -c2 -g1 -K2 -r2 --smpiargs="-disable_gpu_hooks" python3 $PWD/osu-lat-gpu-charm.py +ppn 1 +pemap L0,4 $((1<<10)) $((1<<23)) 1000 100 0 1
    charm4py_cmd_small = ['python3',
                          '/ccs/home/zanef2/charm-mpi-compare/microbench/osu-lat-gpu-charm.py',
                          '1', '1024', '1000', '500', '0', '1'
                          ]
    charm4py_cmd_large = ['python3',
                          '/ccs/home/zanef2/charm-mpi-compare/microbench/osu-lat-gpu-charm.py',
                          '2048', '4194304', '1000', '500', '0', '1'
                          ]


    mpi4py_cmd_small = ['python3',
                        '/ccs/home/zanef2/charm-mpi-compare/microbench/osu-lat-gpu-mpi.py',
                        '1', '1024', '1000', '500', '0', '1'
                          ]
    mpi4py_cmd_large = ['python3',
                        '/ccs/home/zanef2/charm-mpi-compare/microbench/osu-lat-gpu-mpi.py',
                        '2048', '4194304', '1000', '500', '0', '1'
                          ]

    mpi_cmd = ['/ccs/home/zanef2/osu-micro-benchmarks-5.8/build/install/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency', 'D', 'D'
               ]
    charm_cmd = ['/ccs/home/zanef2/charm_lat/latency', '-z']

    mpirun_base = ['-np', '2']
    jsrun_base = ['-n2', '-a1', '-c1', '-g1', '-K2', '-r2', '--smpiargs="-disable_gpu_hooks"']

    mpirun = sh.Command('mpirun')
    jsrun = sh.Command('jsrun')

    mpirun_args = (*mpirun_base, *mpirun_args)

    mpi4py_small = mpirun.bake(*mpirun_args, *mpi4py_cmd_small)
    mpi4py_large = mpirun.bake(*mpirun_args, *mpi4py_cmd_large)
    mpi4py_small._output_f = open('mpi4py_intranode_gpu_lat.csv', 'w')
    mpi4py_large._output_f = mpi4py_large._output_f

    charm4py_small = jsrun.bake(*charm4py_cmd_small, *charm_args)
    charm4py_large = jsrun.bake(*charm4py_cmd_large, *charm_args)
    charm4py_small._output_f = open('charm4py_intranode_gpu_lat.csv', 'w')
    charm4py_large._output_f = small_intra._output_f

    charm_intra = jsrun.bake(*charm_cmd, *charm_args)
    charm_intra._output_f = open('charm_intranode_gpu_lat.csv', 'w')

    mpi_intra = mpirun.bake(*mpirun_args, *mpi_cmd)
    mpi_intra._output_f = open('mpi_intranode_gpu_lat.csv', 'w')


    cmds = [
        (mpi4py_small, mpi4py_large),
        (charm4py_small, charm4py_large),
        charm_intra,
        mpi_intra
    ]

    for i in range(10):
        random.shuffle(cmds)
        for idx, c in enumerate(cmds):
            t_start = time.time()
            if type(c) == tuple:
                cmd_str = str(c[0])
                print(f"Executing command: {c[0]}")
                # flush because writing to the file behaves differently than
                # redirecting to it, can reorder the output
                c[0]._output_f.write(f"# {cmd_str}\n")
                c[0]._output_f.flush()
                c[0](_out=c[0]._output_f, _err=c[0]._output_f)

                cmd_str = str(c[1])
                c[1]._output_f.write(f"# {c[1]md_str}\n")
                c[1]._output_f.flush()
                c[1](_out=c[1]._output_f, _err=c[1]._output_f)
                t_end = time.time()

            else:
                cmd_str = str(c)
                print(f"Executing command: {c}")
                # flush because writing to the file behaves differently than
                # redirecting to it, can reorder the output
                c._output_f.write(f"# {cmd_str}\n")
                c._output_f.flush()
                c(_out=c._output_f, _err=c._output_f)
                t_end = time.time()
            print(f"Command {(i*len(cmds))+idx+1} of {10*len(cmds)} completed in {t_end - t_start}s.")

    for c in cmds:
        c._output_f.close()

if __name__ == '__main__':
    main()
