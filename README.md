# Shared Memory Architecture Matrix Relaxation
Shared memory (`pthread`) method for relaxing large matrices on the HPC cluster at University of Bath, using SLURM with jobs submitted in varying configurations.

The relaxation technique offers a solution of differential equations. This is done by
having an array of values and repeatedly replacing a value with the average of its four neighbours; excepting boundary values, which remain at fixed values. This is repeated until all values settle down to within a given precision.

This technique was implemented as part of [CM30225](http://people.bath.ac.uk/masrjb/CourseNotes/cm30225.html): *Parallel Computing*.

- Compile with: `gcc -std=gnu99 -Wall -Wextra -Wconversion -pthread sharedrelax.c -o sharedrelax`
- Run with: `./sharedrelax [args]`

Where `[args]` are:

    -d (Integer, default=100) The matrix dimensions    
    -t (Integer, default=1) The number of threads   
    -p (Double, default=0.5) The precision threshold   
    -v (No specifier, default=false) Enable verbose mode, do not use while performance testing
    -i (No specifier, default=false) Enable info mode, use while correctness testing