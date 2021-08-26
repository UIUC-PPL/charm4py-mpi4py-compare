import sh
import sys
import random
import time

N_TRIALS = 5
N_ITERS = 100
def main():
    # mpirun -np 12 python3 stencil-mpi.py 100 24576 24576
    # mpirun -np 12 python3 stencil-charm.py 12 100 24576 24576 +no_isomalloc_sync

    basedir = '/home1/08302/tg876011/charm-mpi-compare/stencil'
    mpi_file = f'{basedir}/stencil-mpi.py'
    charm_file = f'{basedir}/stencil-charm.py'
    nprocs = [6,12, 24, 48, 96, 192, 384, 768]
    base_dim = (24576, 24576)

    # 5 trials
    # in each trial, go through each of n procs and run a command for charm, mpi
    # 5 * 7 * 2 = 70 commands
    mpirun = sh.mpirun
    outfile = open('strong_scaling_stdout.txt', 'w')

    for trial in range(N_TRIALS):
        cmds = list()
        for np in nprocs:
            common = [str(N_ITERS),
                      str(base_dim[0]),
                      str(base_dim[1]),
                      'strong_scaling_results'
                      ]

            charm_base = ['python3',
                          charm_file,
                          str(np),
                          *common
                          ]

            charm_suffix = ['+no_isomalloc_sync']
            mpi_base = ['python3',
                        mpi_file,
                        *common
                        ]

            np_base = ['-np', str(np)]

            mpi_cmd = mpirun.bake(*np_base,
                                  *mpi_base
                                  )
            charm_cmd = mpirun.bake(*np_base,
                                    *charm_base,
                                    *charm_suffix
                                    )
            cmds.append(charm_cmd)
            cmds.append(mpi_cmd)
        random.shuffle(cmds)

        for c in cmds:
            tst = time.time()
            print(f"Executing {str(c)}")
            c(_out=outfile)
            tend = time.time()
            print(f"Command took {tend-tst}s")
    outfile.close()


if __name__ == '__main__':
    main()
