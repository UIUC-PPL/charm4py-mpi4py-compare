import sh
import sys
import random
import time
import sys

N_TRIALS = 1

def main():
    # mpirun -np 48 python3 pic-mpi.py -c 48 -p 16 -d 4 --mode GEOMETRIC --init_parameters 0.999 -s 5998 -i 6 -n 6400000 -k 0 -m 1 +no_isomalloc_sync

    basedir = '/home1/08302/tg876011/charm-mpi-compare/pic'
    charm_file = f'{basedir}/pic-charm.py'
    mpi_file = f'{basedir}/pic-mpi.py'
    odf = 8
    lbp = 80
    core_counts = [24, 48, 96, 192, 384, 768]

    # 10 trials
    mpirun = sh.mpirun
    outfile = open(f'pic_strong_scaling_{sys.argv[1]}.txt', 'w')
    charm_base = ['python3', 
                  f'{charm_file}',
		  ]

    mpi_base = ['python3', 
                f'{mpi_file}',
		]

    common = [
              '-p', '5999',
              '-i', '1000',
              '-n', '600000',
              '-k', '0',
              '-m', '1',
              '--init_parameters', '0.999',
              '-s', '5998',
              '--mode', 'GEOMETRIC',
              # mpi4py ignores these parameters
              '-d', str(lbp)
    ]
    suffix = ['+no_isomalloc_sync', '+balancer', 'TreeLB']

    for trial in range(N_TRIALS):
        cmds = list()
        for c in core_counts:
                np_base = ['-np', str(c)]
                charm_cmd = mpirun.bake(*np_base,
                                        *charm_base,
				        *common,
                                        '-c', str(odf*c),
                                        *suffix
                                    )
                mpi_cmd = mpirun.bake(*np_base,
				      *mpi_base,
				      *common
                                    )
                cmds.append(charm_cmd)
                cmds.append(mpi_cmd)
        random.shuffle(cmds)

        for c in cmds:
            tst = time.time()
            print(f"Executing {str(c)}")
            print(f"# {str(c)}", file=outfile)
            outfile.flush()
            c(_out=outfile)
            tend = time.time()
            print(f"Command took {tend-tst}s")
    outfile.close()


if __name__ == '__main__':
    main()
