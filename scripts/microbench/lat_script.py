import sh
import sys
import random
import time


def main():
    intra_mpirun = ['--map-by', 'core', '--bind-to', 'core', '--report-bindings']
    inter_mpirun = ['--map-by', 'socket', '--bind-to', 'core', '--report-bindings']

    intra_charmrun = ['+pemap', 'L0,2', '+no_isomalloc_sync']
    inter_charmrun = ['+pemap', 'L0,1', '+no_isomalloc_sync']

    charm4py_intra_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-charm.py',
                          '1', '4194304', '64', '1000', '500', '0',
                          'charm_lat_intrasocket'
                          ]

    # On stampede Core 1 is on NUMA node 1
    charm4py_inter_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-charm.py',
                          '1', '4194304', '1000', '500', '0',
                          'charm_lat_intersocket'
                          ]


    mpi4py_intra_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-mpi.py',
                        '1', '4194304', '1000', '500', '0',
                        'mpi_lat_intrasocket'
                        ]
    mpi4py_inter_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-mpi.py',
                        '1', '4194304', '1000', '500', '0',
                        'mpi_lat_intersocket'
                        ]

    mpirun_base = ['-np','2']
    srun_base = ['--ntasks=2']

    mpirun = sh.Command('mpirun')
    srun = sh.Command('srun')

    intra_mpirun_args = (*mpirun_base, *intra_mpirun)
    inter_mpirun_args = (*mpirun_base, *inter_mpirun)

    mpi4py_intra = mpirun.bake(*intra_mpirun_args, *mpi4py_intra_cmd)
    mpi4py_intra._output_f = open('mpi4py_intrasocket_lat.csv', 'w')

    mpi4py_inter = mpirun.bake(*inter_mpirun_args, *mpi4py_inter_cmd)
    mpi4py_inter._output_f = open('mpi4py_intersocket_lat.csv', 'w')

    charm_base = srun.bake(*srun_base)

    charm4py_intra = charm_base.bake(*charm4py_intra_cmd, *intra_charmrun)
    charm4py_intra._output_f = open('charm4py_intrasocket_lat.csv', 'w')

    charm4py_inter = charm_base.bake(*charm4py_inter_cmd, *inter_charmrun)
    charm4py_inter._output_f = open('charm4py_intersocket_lat.csv', 'w')


    cmds = [
        mpi4py_intra, mpi4py_inter,
        charm4py_inter, charm4py_intra
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
