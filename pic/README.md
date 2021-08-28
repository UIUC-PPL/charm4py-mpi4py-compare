Note: This library uses the random number generator from the C Parralel Research Kernels, and before running this kernel you must first build the shared library:
```
gcc ./random_draw.c -O3 -lm -fPIC -shared -o librandomdraw.so
```

Then the kernel can be run in the following manner:
```
mpirun -np 9 python3 pic-mpi.py -p 9 -d 2 --mode GEOMETRIC --init_parameters 0.99 -s 300 -i 10 -n 500000 -k 0 -m 1 -v --output out_f.csv --add_datetime
```

