import sh
import sys
import random
import time
import os


def main():
    os.putenv('I_MPI_EAGER_THRESHOLD','8192')
    intra_mpirun = ['--map-by', 'core', '--bind-to', 'core' ]
    inter_mpirun = ['--map-by', 'socket', '--bind-to', 'core']

    intra_charmrun = ['+pemap', 'L0,2', '+no_isomalloc_sync']
    inter_charmrun = ['+pemap', 'L0,1', '+no_isomalloc_sync']

    charm4py_intra_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-charm.py',
                          '1', '4194304', '1000', '500', '0'
                          ]

    # On stampede Core 1 is on NUMA node 1
    charm4py_inter_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-charm.py',
                          '1', '4194304', '1000', '500', '0'
                          ]


    mpi4py_intra_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-mpi.py',
                        '1', '4194304', '1000', '500', '0'
                        ]
    mpi4py_inter_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-mpi.py',
                        '1', '4194304', '1000', '500', '0'
                        ]
    mpi_inter_cmd = ['/home1/08302/tg876011/osu-micro-benchmarks/mpi/pt2pt/osu_latency']
    mpi_intra_cmd = ['/home1/08302/tg876011/osu-micro-benchmarks/mpi/pt2pt/osu_latency']

    charm_inter_cmd = ['/home1/08302/tg876011/charm_lat/latency']
    charm_intra_cmd = ['/home1/08302/tg876011/charm_lat/latency']

    mpirun_base = ['-np', '2']

    mpirun = sh.Command('mpirun')

    intra_mpirun_args = (*mpirun_base, *intra_mpirun)
    inter_mpirun_args = (*mpirun_base, *inter_mpirun)

    mpi4py_intra = mpirun.bake(*intra_mpirun_args, *mpi4py_intra_cmd)
    mpi4py_intra._output_f = open('mpi4py_intrasocket_lat.csv', 'w')

    mpi4py_inter = mpirun.bake(*inter_mpirun_args, *mpi4py_inter_cmd)
    mpi4py_inter._output_f = open('mpi4py_intersocket_lat.csv', 'w')

    charm4py_intra = mpirun.bake(*charm4py_intra_cmd, *intra_charmrun)
    charm4py_intra._output_f = open('charm4py_intrasocket_lat.csv', 'w')

    charm4py_inter = mpirun.bake(*charm4py_inter_cmd, *inter_charmrun)
    charm4py_inter._output_f = open('charm4py_intersocket_lat.csv', 'w')

    charm_inter = mpirun.bake(*charm_inter_cmd, *inter_charmrun)
    charm_inter._output_f = open('charm_intersocket_lat.csv', 'w')

    charm_intra = mpirun.bake(*charm_intra_cmd, *intra_charmrun)
    charm_intra._output_f = open('charm_intrasocket_lat.csv', 'w')

    mpi_inter = mpirun.bake(*inter_mpirun_args, *mpi_inter_cmd)
    mpi_inter._output_f = open('mpi_intersocket_lat.csv', 'w')

    mpi_intra = mpirun.bake(*intra_mpirun_args, *mpi_intra_cmd)
    mpi_intra._output_f = open('mpi_intrasocket_lat.csv', 'w')


    cmds = [
        mpi4py_intra, mpi4py_inter,
        charm4py_inter, charm4py_intra,
        charm_inter, charm_intra,
        mpi_inter, mpi_intra
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
