import sh
import sys
import random
import time


def main():
    inter_mpirun = ['--map-by', 'core', '--bind-to', 'core' ]

    inter_charmrun = ['+pemap', 'L0', '+no_isomalloc_sync']

    charm4py_inter_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-charm.py',
                          '1', '4194304', '1000', '500', '0'
                          ]

    mpi4py_inter_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-lat-mpi.py',
                        '1', '4194304', '1000', '500', '0'
                        ]
    mpi_inter_cmd = ['/home1/08302/tg876011/osu-micro-benchmarks/mpi/pt2pt/osu_latency']

    charm_inter_cmd = ['/home1/08302/tg876011/charm_lat/latency']

    mpirun_base = ['-np', '2']

    mpirun = sh.Command('mpirun')

    inter_mpirun_args = (*mpirun_base, *inter_mpirun)

    mpi4py_inter = mpirun.bake(*inter_mpirun_args, *mpi4py_inter_cmd)
    mpi4py_inter._output_f = open('mpi4py_internode_lat.csv', 'w')

    charm4py_inter = mpirun.bake(*charm4py_inter_cmd, *inter_charmrun)
    charm4py_inter._output_f = open('charm4py_internode_lat.csv', 'w')

    charm_inter = mpirun.bake(*charm_inter_cmd, *inter_charmrun)
    charm_inter._output_f = open('charm_internode_lat.csv', 'w')

    mpi_inter = mpirun.bake(*inter_mpirun_args, *mpi_inter_cmd)
    mpi_inter._output_f = open('mpi_internode_lat.csv', 'w')


    cmds = [
        charm4py_inter,
        charm_inter, mpi4py_inter,
        mpi_inter
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
