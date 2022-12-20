This is the svds-C algorithm compiled and linked with MKL and OpenMP.

Firstly, the path of MKL in makefile should be modified to the path you defined before.

At last, use following commands to run the test case (The number of threads can be modified by user):
    "make"
    "OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ./svdstest"

In svds.c, the main functions are:
    svds_C(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k);
    svds_C_opt(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double tol, int maxbasis, int maxiter);
    svds_C_dense(mat* A, mat **Uk, mat **Sk, mat **Vk, int k);
    svds_C_dense_opt(csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double tol, int maxbasis, int maxiter);
    
