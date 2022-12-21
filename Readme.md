# svds-C (High-performance Program for Computing Truncated SVD)

## Author: Xu Feng (Tsinghua University)

svds-C is an open-source software for computing truncated SVD of large matrix.

* [Background](#background)
* [Software](#software)
  + [Prerequisites](#prerequisites)
  + [Modules of the software](#modules-of-the-software)
  + [Examples of execution](#examples-of-execution)

* [References](#references)

## Background

Truncated singular value decomposition (SVD) is an important tool for many computational physics problems. However, the most efficient algorithm of truncated singular value decomposition for large matrix has only an implementation in Matlab. Other programs for truncated SVD have issues of stability or efficiency on parallel computing.

We develop a parallel C program named svds-C, which re-implements the svds in Matlab based on Lanczos bidiagonalization process with augmented restarting scheme. To make the svds-C program running most efficiently on a common computer with only CPU, we employ MKL [1] or OpenBLAS [2] to enable high-performance multiple-thread computing. Besides, several careful treatments are imposed in the implementation for better performance on time and memory cost. Experimental results show that the svds-C exhibits better parallel efficiency and runtime advantage over svds and other programs for truncated SVD. The svds-C program is open-sourced.

## Software

### Prerequisites

svds-C has been written in C with MKL or OpenBLAS.

We provide two versions of svds-C program for Intel CPU and AMD CPU respectively. The Intel version requires the external libraries of MKL. The AMD version requires the external libraries of OpenBLAS. Before using this program, one should download and install MKL in [1] or OpenBLAS in [2] to complete the preparation. Besides, OpenMP is used in both versions, make sure OpenMP is available in the system.

### Modules of the software

**The version for Intel CPU**:

​        Codes are merged in "Codes/Intel/".

​        **svds.h**: contains the interfaces of svds-C for accurate truncated SVD of sparse matrix and dense matrix.

​        **svds.c**: implementations of svds-C. "svds_C" is the C version of svds in Matlab for truncated SVD with default options while and "svds_C_opt" is the version with user's options, these two functions are designed for sparse matrix in CSR format. While, "svds_C_dense" and "svds_C_dense_opt" are designed for dense matrix.

​        **matrix_vector_functions_intel_mkl.c**, **matrix_vector_functions_intel_mkl_ext.c**: contain the basic functions for matrix-matrix, matrix-vector and vector-vector operations implemented with MKL and OpenMP.

**The version for AMD CPU**:

​        Codes are merged in "Codes/AMD/".

​        **svds.h**: contains the interfaces of svds-C for accurate truncated SVD of sparse matrix and dense matrix.

​        **svds.c**: implementations of svds-C. "svds_C" is the C version of svds in Matlab for truncated SVD with default options while and "svds_C_opt" is the version with user's options, these two functions are designed for sparse matrix in CSR format. While, "svds_C_dense" and "svds_C_dense_opt" are designed for dense matrix.

​        **matrix_funs.c**: contains the the basic functions for matrix-matrix, matrix-vector and vector-vector operations implemented with OpenBLAS and OpenMP.

**Interfaces of svds-C**:

​        **svds_C(A, U, S, V, k)**: compute the rank-$k$ truncated SVD of the sparse matrix $\mathbf{A}$ with default settings. $\mathbf{U}$, $\mathbf{S}$ and $\mathbf{V}$ are the computed left singular vectors, singular values and right singular vectors, respectively.

​        **svds_C_opt(A, U, S, V, k, tol, t, r)**: compute the rank-$k$ truncated SVD of the sparse matrix $\mathbf{A}$ with user's settings. Where **tol** represents the relative residual tolerance, **t** represents the size of the subspace, while **r** represents the maximum of restarting.

​        **svds_C_dense(A, U, S, V, k)**: compute the rank-$k$ truncated SVD of the dense matrix $\mathbf{A}$ with default settings. 

​        **svds_C_dense_opt(A, U, S, V, k, tol, t, r)**: compute the rank-$k$ truncated SVD of the dense matrix $\mathbf{A}$ with user's settings.

### Examples of execution

All the files for testing are merged in "Tests/"

**The version for Intel CPU**:

​        Before compilation, the MKL in oneAPI in [1] should be installed and the path of the libraries of MKL and oneAPI should be modified in "makefile" to make sure those libraries can be found.

​        Use command "make TYPE_CPU=Intel" to compile the test program for Intel CPU, and run "./svdstest" to test svds-C.

**The version for AMD CPU**:

​        Before compilation, the OpenBLAS in [2] should be installed and the path of OpenBLAS should be modified in "makefile" to make sure OpenBLAS can be found.

​        Use command "make TYPE_CPU=AMD" to compile the test program for Intel CPU, and run "./svdstest" to test svds-C.

**Test cases**:

​        In the executable "svdstest", we prepare a sparse matrix SNAP in size of 82168 x 82168 [3], and a dense Gaussian random matrix in size of  2000 x 1000 to validate svds-C. 

## References

[1] Intel oneAPI Math Kernel Library, https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html (2021).

[2] X. Zhang, OpenBLAS: An optimized BLAS library, http://www.openblas.net/ (2022). 

[3] J. Leskovec, A. Krevl, SNAP datasets: Stanford large network dataset collection, http://snap.stanford.edu/data (Jun. 2014) 
