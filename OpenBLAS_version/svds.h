#pragma once

#include "matrix_funcs.h"

void svds_C(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k);

void svds_C_opt(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter);

void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);

void svds_C_dense_opt(mat *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter);
