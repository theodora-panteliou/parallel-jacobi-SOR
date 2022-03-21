#!/bin/bash

# JobName #
#PBS -N myJob

# Which Queue to use, DO NOT CHANGE #
#PBS -q N10C80

# Max VM size #
#PBS -l pvmem=2G

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:02:00

# How many nodes, cpus/node, mpiprocs/node and threds/mpiprocess
# Example 2 nodes with 8 cpus/node, 2 mpiprocs/node and 4 threads/mpiproc
#PBS -l select=8:ncpus=8:mpiprocs=8:mem=16400000kb
#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun mpijac.x < input

