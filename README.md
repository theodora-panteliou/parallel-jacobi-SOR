# Parallel Jacobi with Successive Overelaxation
Assignment for the course "Parallel Systems" in 2020-2021. 
Jacobi with successive over-relaxation parallel implementations with MPI, MPI+OpenMP, CUDA to solve the Poisson's equation:

![image](https://user-images.githubusercontent.com/60042402/159339800-ab4cbf9e-3a3e-4065-a9e1-32a2124a75ec.png)

## ParallelMPI
By using the MPI 2-d cartesian topology the matrix is divided to smaller ones, that exchange informations with their neighbors in order to do the computations.

## HybridMPI (MPI+OpenMP)
Every MPI process creates OpenMP threads and `#omp for` is used to distribute the iterations to the threads. Only the main thread runs MPI instructions.

## CUDA
Implemented for 1 and 2 GPUs. The main loop now calls a function that runs on CUDA. In the case of two gpus, 2 openMP threads are created that each one runs on a different GPU device.

## References
* https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
* Code and slides from the course "Parallel Systems" at DIT, NKUA.

## Authors
* [Dora Panteliou](https://github.com/dora-jpg)
* [Ioannis Karakolis](https://github.com/sdi1800065)
