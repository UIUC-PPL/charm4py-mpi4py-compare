import sh
import sys
import random
import time


def main():
    intra_mpirun = ['--map-by', 'core', '--bind-to', 'core', '--report-bindings']
    inter_mpirun = ['--map-by', 'socket', '--bind-to', 'core', '--report-bindings']

    intra_charmrun = ['+pemap', 'L0,2']
    inter_charmrun = ['+pemap', 'L0,1']

    charm4py_intra_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-bw-charm.py',
                          '1', '4194304', '64', '1000', '100',
                          'charm_bw_intrasocket'
                          ]

    # On stampede Core 1 is on NUMA node 1
    charm4py_inter_cmd = ['python3',
                          '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-bw-charm.py',
                          '1', '4194304', '64', '1000', '100',
                          'charm_bw_intersocket'
                          ]

    charm_intra_cmd = ['/home1/08302/tg876011/charm_bw/bandwidth']
    charm_inter_cmd = ['/home1/08302/tg876011/charm_bw/bandwidth']

    mpi4py_intra_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-bw-mpi.py',
                        '1', '4194304', '64', '1000', '100',
                        'mpi_bw_intrasocket'
                        ]
    mpi4py_inter_cmd = ['python3',
                        '/home1/08302/tg876011/charm-mpi-compare/microbenchmarks/osu-bw-mpi.py',
                        '1', '4194304', '64', '1000', '100',
                        'mpi_bw_intersocket'
                        ]

    mpi_intra_cmd = ['/home1/08302/tg876011/osu-micro-benchmarks/mpi/pt2pt/osu_bw']
    mpi_inter_cmd = ['/home1/08302/tg876011/osu-micro-benchmarks/mpi/pt2pt/osu_bw']

    mpirun_base = ['-np','2']
    srun_base = ['--ntasks=2']

    mpirun = sh.Command('mpirun')
    srun = sh.Command('srun')

    intra_mpirun_args = (*mpirun_base, *intra_mpirun)
    inter_mpirun_args = (*mpirun_base, *inter_mpirun)

    mpi4py_intra = mpirun.bake(*intra_mpirun_args, *mpi4py_intra_cmd)
    mpi4py_intra._output_f = open('mpi4py_intrasocket.csv', 'w')

    mpi4py_inter = mpirun.bake(*inter_mpirun_args, *mpi4py_inter_cmd)
    mpi4py_inter._output_f = open('mpi4py_intersocket.csv', 'w')

    mpi_intra = mpirun.bake(*intra_mpirun_args, *mpi_intra_cmd)
    mpi_intra._output_f = open('mpi_intrasocket.csv', 'w')

    mpi_inter = mpirun.bake(*inter_mpirun_args, *mpi_inter_cmd)
    mpi_inter._output_f = open('mpi_intersocket.csv', 'w')

    charm_base = srun.bake(*srun_base)

    charm4py_intra = charm_base.bake(*charm4py_intra_cmd, *intra_charmrun)
    charm4py_intra._output_f = open('charm4py_intrasocket.csv', 'w')

    charm4py_inter = charm_base.bake(*charm4py_inter_cmd, *inter_charmrun)
    charm4py_inter._output_f = open('charm4py_intersocket.csv', 'w')

    charm_intra = charm_base.bake(*charm_intra_cmd, *intra_charmrun)
    charm_intra._output_f = open('charm_intrasocket.csv', 'w')

    charm_inter = charm_base.bake(*charm_inter_cmd, *inter_charmrun)
    charm_inter._output_f = open('charm_intersocket.csv', 'w')


    cmds = [
        mpi4py_intra, mpi4py_inter,
        mpi_intra, mpi_inter,
        charm4py_inter, charm4py_intra,
        charm_inter, charm_intra
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
