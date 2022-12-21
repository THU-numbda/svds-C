#pragma once

#include "mkl.h"
#include "matrix_vector_functions_intel_mkl.h"
#include "matrix_vector_functions_intel_mkl_ext.h"


void svds_C(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k);

void svds_C_opt(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter);

void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);

void svds_C_dense_opt(mat *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter);
