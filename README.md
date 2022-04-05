# parallel-jacobi-SOR
Jacobi with successive over-relaxation parallel implementations with MPI, MPI+OpenMP, CUDA to solve the Poisson's equation:

![image](https://user-images.githubusercontent.com/60042402/159339800-ab4cbf9e-3a3e-4065-a9e1-32a2124a75ec.png)

## ParallelMPI
By using the MPI 2-d cartesian topology the matrix is divided to smaller ones, that exchange informations with their neighbors in order to do the computations.

## HybridMPI (MPI+OpenMP)
Every MPI process creates OpenMP threads and `#omp for` is used to distribute the iterations to the threads. Only the main thread runs MPI instructions.

## CUDA
