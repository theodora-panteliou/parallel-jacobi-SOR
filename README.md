# parallel-jacobi-SOR
Jacobi with successive over-relaxation parallel implementations with MPI, MPI+OpenMP, CUDA to solve the Poisson's equation:

![image](https://user-images.githubusercontent.com/60042402/159339800-ab4cbf9e-3a3e-4065-a9e1-32a2124a75ec.png)

## Sequential
Improved the execution time of the initial serial jacobi program. We made the functions inline and 

## ParallelMPI
We applied the foster methodology to divide the data By using the MPI 2-d cartesian topology we divided the matrix to smaller ones, that exchange informations with their neighbors in order to do the computations. In more detail, w

## HybridMPI (MPI+OpenMP)

## CUDA
