# Project Overview

This project showcases several CUDA programs designed to demonstrate a range of parallel computing concepts using the CUDA programming model. Each subdirectory contains a distinct CUDA program, each targeting a specific operation or technique in parallel computing.

## Directory Structure

- `HelloWorld/`: A basic CUDA program that demonstrates executing a kernel on the GPU by printing "Hello World".
- `MatrixMul/`: Includes CUDA programs for matrix multiplication with two versions: one for standard matrix multiplication and another that utilizes tiled matrix multiplication for enhanced performance.
- `SumReduction/`: Contains a CUDA program focused on sum reduction, showcasing efficient parallel reduction techniques on the GPU.
- `VectorAdd/`: Features a program for vector addition, illustrating simple parallel arithmetic operations on the GPU.
- `SpMV/`: Features a program for sparse CSR matrix vector multiplication.
- `MD/`: Features a program for simple molecular dynamics of a set of particles with a constant temperature interacting with each other through Lennard-Jones potential.

## Building and Running the Programs

Each program directory is equipped with a `Makefile` for straightforward compilation. To compile a program, navigate to the specific directory of the program and execute the `make` command. For example, to compile the `VectorAdd` program, you would use:

make

After compiling, run the generated binary to execute the program. For instance, to run `VectorAdd`, execute:

./VectorAdd

## About the CUDA Programs

These CUDA programs leverage the CUDA programming model to execute computations on NVIDIA GPUs. They cover various CUDA concepts, including kernel execution, memory management, synchronization, and error handling.

- `MatrixMul` demonstrates matrix multiplication on the GPU, highlighting the efficient use of shared memory and thread synchronization.
- `HelloWorld` provides a simple introduction to kernel execution on the GPU.
- `SumReduction` and `VectorAdd` showcase parallel algorithms for reduction and vector addition, respectively, emphasizing efficient parallel data processing techniques on the GPU.
- `MD`: This directory contains a program for an n-body simulation, a type of simulation that calculates the interaction between multiple particles in a system. The particles interact with each other through the Lennard-Jones potential. The program uses an optimized sum reduction kernel for efficient parallel processing on the GPU. To visualize the results of the simulation, you will need to use OVITO, an open-source visualization and analysis software.


## Note

Please ensure the CUDA toolkit is installed and correctly configured on your system before attempting to build and run these programs. This includes setting up necessary environment variables and verifying your hardware supports CUDA.
