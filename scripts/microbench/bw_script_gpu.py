import sh
import sys
import random
import time
import os
import sys


def main():
    output_base = '/gpfs/alpine/scratch/zanef2/csc357'
    hostname = sys.argv[1]
    os.putenv('UCX_RNDV_THRESH', '131072')
    os.putenv('UCX_MEMTYPE_CACHE', 'n')
    mpirun_args = ['-H', f'{hostname}:2', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH',
                   '-x', 'UCX_RNDV_THRESH=131072', '-x', 'UCX_MEMTYPE_CACHE=n',
                   ]

    charm_args = ['+ppn', '1', '+pemap', 'L0,4']

    # UCX_RNDV_THRESH=131072 UCX_MEMTYPE_CACHE=n jsrun -n2 -a1 -c2 -g1 -K2 -r2 --smpiargs="-disable_gpu_hooks" python3 $PWD/osu-bw-gpu-charm.py +ppn 1 +pemap L0,4 $((1<<10)) $((1<<23)) 1000 100 0 1
    charm4py_cmd_small = ['python3',
                          '/ccs/home/zanef2/charm-mpi-compare/microbenchmarks/osu-bw-gpu-charm.py',
                          '1', '1024', '64', '1000', '500', '1'
                          ]
    charm4py_cmd_large = ['python3',
                          '/ccs/home/zanef2/charm-mpi-compare/microbenchmarks/osu-bw-gpu-charm.py',
                          '2048', '4194304', '64', '1000', '500', '1'
                          ]


    mpi4py_cmd_small = ['python3',
                        '/ccs/home/zanef2/charm-mpi-compare/microbenchmarks/osu-bw-gpu-mpi.py',
                        '1', '1024', '64', '1000', '500'
                          ]
    mpi4py_cmd_large = ['python3',
                        '/ccs/home/zanef2/charm-mpi-compare/microbenchmarks/osu-bw-gpu-mpi.py',
                        '2048', '4194304', '64', '1000', '500'
                          ]

    mpi_cmd = ['/ccs/home/zanef2/osu-micro-benchmarks-5.8/build/install/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw', 'D', 'D'
               ]
    charm_cmd = ['/ccs/home/zanef2/charm_bw/bandwidth']

    mpirun_base = ['-np', '2']
    jsrun_base = ['-n2', '-a1', '-c1', '-g1', '-K2', '-r2', '--smpiargs="-disable_gpu_hooks"']

    mpirun = sh.Command('mpirun')
    jsrun = sh.Command('jsrun')
    jsrun = jsrun.bake(*jsrun_base)

    mpirun_args = (*mpirun_base, *mpirun_args)

    mpi4py_small = mpirun.bake(*mpirun_args, *mpi4py_cmd_small)
    mpi4py_large = mpirun.bake(*mpirun_args, *mpi4py_cmd_large)
    mpi4py_small._output_f = open(f'{output_base}/mpi4py_intranode_gpu_bw.csv', 'w')
    mpi4py_large._output_f = mpi4py_small._output_f

    charm4py_small = jsrun.bake(*charm4py_cmd_small, *charm_args)
    charm4py_large = jsrun.bake(*charm4py_cmd_large, *charm_args)
    charm4py_small._output_f = open(f'{output_base}/charm4py_intranode_gpu_bw.csv', 'w')
    charm4py_large._output_f = charm4py_small._output_f

    charm_intra = jsrun.bake(*charm_cmd, *charm_args)
    charm_intra._output_f = open(f'{output_base}/charm_intranode_gpu_bw.csv', 'w')

    mpi_intra = mpirun.bake(*mpirun_args, *mpi_cmd)
    mpi_intra._output_f = open(f'{output_base}/mpi_intranode_gpu_bw.csv', 'w')


    cmds = [
        mpi4py_small, mpi4py_large,
        charm4py_small, charm4py_large,
        charm_intra,
        mpi_intra
    ]

    for i in range(10):
        random.shuffle(cmds)
        for idx, c in enumerate(cmds):
            t_start = time.time()
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
