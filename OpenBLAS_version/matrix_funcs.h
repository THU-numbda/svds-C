#pragma once

#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

typedef struct {
    int nrows, ncols;
    double * d;
} mat;


typedef struct {
    int nrows;
    double * d;
} vec;

typedef struct {
    int nrows, ncols;
    long long nnz; // number of non-zero element in the matrix.
    long long capacity; // number of possible nnzs.
    double *values;
    int *rows, *cols;
} mat_coo;

typedef struct {
    long long nnz;
    int nrows, ncols;
    double *values;
    int *cols;
    int *pointerB, *pointerE;
} mat_csr;


void initialize_random_vector(vec *M);

void initialize_random_matrix_double(mat *M);

mat * matrix_new(int nrows, int ncols);

vec * vector_new(int nrows);

void matrix_delete(mat *M);

void vector_delete(vec *v);

void matrix_print(mat * M);

void matrix_copy(mat *D, mat *S);

void vector_copy(vec *d, vec *s);

double vector_dot_product(vec *u, vec *v);

void matrix_build_transpose(mat *Mt, mat *M);

double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval);

void matrix_matrix_mult(mat *A, mat *B, mat *C);

void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C);

void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C);

void matrix_vector_mult(mat *M, vec *x, vec *y);

void matrix_transpose_vector_mult(mat *M, vec *x, vec *y);

void matrix_set_element(mat *M, int row_num, int col_num, double val);

double matrix_get_element(mat *M, int row_num, int col_num);

void matrix_get_selected_columns(mat *M, int *inds, mat *Mc);

void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt);

mat_coo* coo_matrix_new(int nrows, int ncols, int capacity);

mat_csr* csr_matrix_new();

void csr_matrix_delete(mat_csr *M);

void csr_matrix_delete(mat_csr *M);

void csr_init_from_coo(mat_csr *D, mat_coo *M);

void coo_matrix_delete(mat_coo *M);

void csr_matrix_vector_mult(mat_csr *A, vec *x, vec *y);

void csr_matrix_transpose_vector_mult(mat_csr *A, vec *x, vec *y);

