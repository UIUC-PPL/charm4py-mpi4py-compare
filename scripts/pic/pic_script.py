import sh
import sys
import random
import time

N_TRIALS = 1

N_CORES=384

def main():
    # mpirun -np 48 python3 pic-mpi.py -c 48 -p 16 -d 4 --mode GEOMETRIC --init_parameters 0.999 -s 5998 -i 6 -n 6400000 -k 0 -m 1 +no_isomalloc_sync

    basedir = '/home1/08302/tg876011/charm-mpi-compare/scripts/pic'
    charm_file = f'{basedir}/pic-charm.py'
    odf = [1,2,4,8,16]
    lbp = [1,2,4,8,16]
    lbp = list(map(lambda x: x*10, lbp))

    # 10 trials
    mpirun = sh.mpirun
    outfile = open('pic_lbfreq_odf_stdout_2trials_2.txt', 'w')
    np_base = ['-np', str(N_CORES)]
    common = ['python3', 
              '/home1/08302/tg876011/charm-mpi-compare/pic/pic-charm.py',
              '-p', '5999',
              '-i', '1000',
              '-n', '6400000',
              '-k', '0',
              '-m', '1',
              '--init_parameters', '0.999',
              '-s', '5998',
              '--mode', 'GEOMETRIC'
    ]
    suffix = ['+no_isomalloc_sync', '+balancer', 'TreeLB']

    for trial in range(N_TRIALS):
        cmds = list()
        for o in odf:
            for lb in lbp:
                this_params = ['-c', str(N_CORES*o),
                               '-d', str(lb)
                ]
                charm_cmd = mpirun.bake(*np_base,
                                    *common,
                                    *this_params,
                                    *suffix
                                    )
                cmds.append(charm_cmd)
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
