import os
import sh
import random
import time
N_TRIALS = 10

def get_nodelist():
    nodes = os.environ.get('LSB_HOSTS').split()[1::]
    nodes = list(set(nodes))
    for i, n in enumerate(nodes):
        nodes[i] = n + ':6'
    return set(nodes)

def main():
    # UCX_MEMTYPE_CACHE=n jsrun -n6 -a1 -c1 -g1 -K3 -r6 --smpiargs="-disable_gpu_hooks" python3 stencil-charm-gpu.py 6 100 76668 76668 /gpfs/alpine/scratch/zanef2/csc357/ +pemap L0,4,8,84,88,92
    # /ccs/home/zanef2/.local/ompi/bin/mpirun -np 6 -H h27n06:6 -x PATH -x LD_LIBRARY_PATH -x UCX_MEMTYPE_CACHE=n python3 stencil-mpi-gpu.py 100 76668 76668 /gpfs/alpine/scratch/zanef2/csc357/2021-09-09_gpu_scaling/weak_scaling
    stdoutfile = '/gpfs/alpine/scratch/zanef2/csc357/2021-09-09_gpu_scaling/weak_scaling_stdout.txt'
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'
    nodelist = get_nodelist()
    print(f"Running on nodes: {','.join(nodelist)}.")

    jsrun_static = ['-a1', '-c1', '-g1', '-K3', '-r6', '--smpiargs="disable_gpu_hooks"']
    mpirun_static = ['-x', 'PATH', '-x', 'LD_LIBRARY_PATH', '-x', 'UCX_MEMTYPE_CACHE=n', '-H', ','.join(nodelist)]
    charm_suffix = ['+pemap', 'L0,4,8,84,88,92']

    node_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    basedim = [73728, 73728]
    niters = '100'
    outdir = '/gpfs/alpine/scratch/zanef2/csc357/2021-09-09_gpu_scaling/weak_scaling'
    common_prefix = ['UCX_MEMTYPE_CACHE=n']

    mpirun_path = '/ccs/home/zanef2/.local/ompi/bin/mpirun'


    jsrun = sh.Command('jsrun').bake(*jsrun_static)
    mpirun = sh.Command(mpirun_path).bake(*mpirun_static)
    outfile = open(stdoutfile, 'w')

    for trial in range(N_TRIALS):
        cmds = list()
        current_dim = basedim.copy()
        for idx, c in enumerate(node_counts):
            if idx > 0:
                if idx % 2:
                    current_dim[0] *= 2
                else:
                    current_dim[1] *= 2

            nprocs = c * 6
            mpi_cmd = mpirun.bake()
            charm_cmd = jsrun.bake(f'-n{str(nprocs)}',
                                   'python3', '/ccs/home/zanef2/charm-mpi-compare/stencil/stencil-charm-gpu.py',
                                   str(nprocs), niters,
                                   str(current_dim[0]), str(current_dim[1]),
                                   outdir,
                                   *charm_suffix
                                   )
            mpi_cmd = mpirun.bake('-np', str(nprocs),
                                  'python3', '/ccs/home/zanef2/charm-mpi-compare/stencil/stencil-mpi-gpu.py',
                                  niters,
                                  str(current_dim[0]), str(current_dim[1]),
                                  outdir
                                  )
            cmds.append(mpi_cmd)
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
