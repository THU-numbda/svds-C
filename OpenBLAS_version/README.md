This is the svds-C algorithm compiled and linked with OpenBLAS and OpenMP.

Firstly, the OpenBLAS should be installed with the correct command as:
    "make USE_OPENMP=1"
    "make install PREFIX=/path/to/your/installation"
Then, the path of OpenBLAS in makefile should be modified to the path you defined before.

At last, use following commands to run the test case (The number of threads can be modified by user):
    "make"
    "OMP_NUM_THREADS=16 ./svdstest"
When svds_C deal with sparse matrix with large size, you may use
    "OMP_NUM_THREADS=16 OMP_STACKSIZE=1G ./svdstest"
to enlarge the stack size of OpenMP.

In svds.c, the main functions are:
    svds_C(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k);
    svds_C_opt(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double tol, int maxbasis, int maxiter);
    svds_C_dense(mat* A, mat **Uk, mat **Sk, mat **Vk, int k);
    svds_C_dense_opt(csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double tol, int maxbasis, int maxiter);
    
