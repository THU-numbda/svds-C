#include <stdio.h>
#include <lapacke.h>
#include <cblas.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_funcs.h"
#include "omp.h"

mat * matrix_new(int nrows, int ncols)
{
    mat *M = malloc(sizeof(mat));
    //M->d = (double*)mkl_calloc(nrows*ncols, sizeof(double), 64);
    long long Size = nrows;
    Size *= ncols;
    M->d = (double*)calloc(Size, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}


/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows)
{
    vec *v = malloc(sizeof(vec));
    //v->d = (double*)mkl_calloc(nrows,sizeof(double), 64);
    v->d = (double*)calloc(nrows,sizeof(double));
    v->nrows = nrows;
    return v;
}


void matrix_delete(mat *M)
{
    //mkl_free(M->d);
    free(M->d);
    free(M);
}


void vector_delete(vec *v)
{
    //mkl_free(v->d);
    free(v->d);
    free(v);
}

void matrix_print(mat * M){
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%.16f  ", val);
        }
        printf("\n");
    }
}

double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval){
    long secs_used, micros_used;
    secs_used=(end_timeval.tv_sec - start_timeval.tv_sec);
    micros_used= ((secs_used*1000000) + end_timeval.tv_usec) - (start_timeval.tv_usec);
    return (micros_used/1e6); 
}

void vector_copy(vec *d, vec *s){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(d,s) private(i) 
    {
    #pragma omp for 
    for(i=0; i<(s->nrows); i++){
        d->d[i] = s->d[i];
    }
    }
}

double vector_dot_product(vec *u, vec *v){
    int i;
    double dotval = 0;
    #pragma omp parallel shared(u,v,dotval) private(i) 
    {
    #pragma omp for reduction(+:dotval)
    for(i=0; i<u->nrows; i++){
        dotval += (u->d[i])*(v->d[i]);
    }
    }
    return dotval;
}

/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}

/* y = M*x ; column major */
void matrix_vector_mult(mat *M, vec *x, vec *y){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemv (CblasColMajor, CblasNoTrans, M->nrows, M->ncols, alpha, M->d, M->nrows, x->d, 1, beta, y->d, 1);
}


/* y = M^T*x ; column major */
void matrix_transpose_vector_mult(mat *M, vec *x, vec *y){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemv (CblasColMajor, CblasTrans, M->nrows, M->ncols, alpha, M->d, M->nrows, x->d, 1, beta, y->d, 1);
}

void initialize_random_vector(vec *M){
    int i,m;
    double val;
    m = M->nrows;
    int N = m;
    srand((unsigned)time(NULL));
    for(i=0;i<N;i++)
        M->d[i] = 1.0*(rand())/RAND_MAX;
}

void initialize_random_matrix_double(mat *M){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;
    int N = m*n;
    srand((unsigned)time(NULL));
    for(i=0;i<N;i++)
        M->d[i] = 1.0*(rand())/RAND_MAX;
}

void vector_set_element(vec *v, int row_num, double val){
    v->d[row_num] = val;
}


double vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}

void matrix_set_element(mat *M, int row_num, int col_num, double val){
    //M->d[row_num*(M->ncols) + col_num] = val;
    long long index = col_num;
    index *= M->nrows;
    index += row_num;
    M->d[index] = val; 
    //M->d[col_num*(M->nrows) + row_num] = val;
}

double matrix_get_element(mat *M, int row_num, int col_num){
    //return M->d[row_num*(M->ncols) + col_num];
    long long index = col_num;
    index *= M->nrows;
    index += row_num;
    return M->d[index];
    //return M->d[col_num*(M->nrows) + row_num];
}

void matrix_build_transpose(mat *Mt, mat *M){
    int i,j;
    for(i=0; i<(M->nrows); i++){
        for(j=0; j<(M->ncols); j++){
            matrix_set_element(Mt,j,i,matrix_get_element(M,i,j)); 
        }
    }
}

void matrix_copy(mat *D, mat *S){
    int i;
    //#pragma omp parallel for
    #pragma omp parallel shared(D,S) private(i) 
    {
    #pragma omp for 
    for(i=0; i<((S->nrows)*(S->ncols)); i++){
        D->d[i] = S->d[i];
    }
    }
}

void initialize_diagonal_matrix(mat *D, vec *data){
    int i;
    #pragma omp parallel shared(D) private(i)
    { 
    #pragma omp parallel for
    for(i=0; i<(D->nrows); i++){
        matrix_set_element(D,i,i,data->d[i]);
    }
    }
}

/* computes SVD: M = U*S*Vt; note Vt = V^T */
void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt){
    int m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    vec * work = vector_new(2*max(3*min(m, n)+max(m, n), 5*min(m,n)));
    vec * svals = vector_new(k);

    LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'S', 'S', m, n, M->d, m, svals->d, U->d, m, Vt->d, k, work->d );

    initialize_diagonal_matrix(S, svals);

    vector_delete(work);
    vector_delete(svals);
}

void matrix_get_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp parallel for
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


void matrix_set_col(mat *M, int j, vec *column_vec){
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp for
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}

void matrix_get_selected_columns(mat *M, int *inds, mat *Mc){
    int i;
    vec *col_vec;
    //printf("%d %d\n", M->ncols, Mc->ncols); 
    #pragma omp parallel shared(M,Mc,inds) private(i,col_vec) 
    {
    #pragma omp parallel for
    for(i=0; i<(Mc->ncols); i++){
        //printf("line:%d\n", i);
        col_vec = vector_new(M->nrows);
        matrix_get_col(M,inds[i],col_vec);
        matrix_set_col(Mc,i,col_vec);
        vector_delete(col_vec);
    }
    }
}


mat_coo* coo_matrix_new(int nrows, int ncols, int capacity) {
    mat_coo *M = (mat_coo*)malloc(sizeof(mat_coo));
    M->values = (double*)calloc(capacity, sizeof(double));
    M->rows = (int*)calloc(capacity, sizeof(int));
    M->cols = (int*)calloc(capacity, sizeof(int));
    M->nnz = 0;
    M->nrows = nrows; M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void coo_matrix_delete(mat_coo *M) {
    free(M->values);
    free(M->cols);
    free(M->rows);
    free(M);
}

mat_csr* csr_matrix_new() {
    mat_csr *M = (mat_csr*)malloc(sizeof(mat_csr));
    return M;
}

void csr_matrix_delete(mat_csr *M) {
    free(M->values);
    free(M->cols);
    free(M->pointerB);
    free(M->pointerE);
    free(M);
}

void csr_init_from_coo(mat_csr *D, mat_coo *M) {
    D->nrows = M->nrows; 
    D->ncols = M->ncols;
    D->pointerB = (int*)malloc(D->nrows*sizeof(int));
    D->pointerE = (int*)malloc(D->nrows*sizeof(int));
    D->cols = (int*)calloc(M->nnz, sizeof(int));
    D->nnz = M->nnz;
    
    D->values = (double*)malloc(M->nnz * sizeof(double));
    memcpy(D->values, M->values, M->nnz * sizeof(double));
    
    int current_row, cursor=0;
    for (current_row = 0; current_row < D->nrows; current_row++) {
        D->pointerB[current_row] = cursor+1;
        while (M->rows[cursor]-1 == current_row) {
            D->cols[cursor] = M->cols[cursor];
            cursor++;
        }
        D->pointerE[current_row] = cursor+1;
    }
}

void csr_matrix_vector_mult(mat_csr *A, vec *x, vec *y) {
    int i;
    int j;
    #pragma omp parallel shared(x,A,y) private(i,j) 
    {
        #pragma omp for
        for(i=0;i<A->nrows;i++)
        {
            y->d[i] = 0;
        }
    
        #pragma omp for
        for(i=0;i<A->nrows;i++)
            for(j=A->pointerB[i];j<A->pointerE[i];j++)
            {
                int t = A->cols[j-1] - 1;
                y->d[i] = y->d[i] + x->d[t]*A->values[j-1];
            }
    }
}

void csr_matrix_transpose_vector_mult(mat_csr *A, vec *x, vec *y) {
    int i;
    int j;

    double ylocal[y->nrows];
    #pragma omp parallel shared(x,A,ylocal) private(i,j) 
    {
        #pragma omp for
        for(i=0;i<A->ncols;i++)
        {
            y->d[i] = 0;
            ylocal[i] = 0;
        }
        
        
        #pragma omp for reduction(+:ylocal)
        for(i=0;i<A->nrows;i++)
        {
            for(j=A->pointerB[i];j<A->pointerE[i];j++)
            {
                int t = A->cols[j-1]-1;
                ylocal[t] = ylocal[t] + x->d[i]*A->values[j-1];
            }
        
        }
    }

    for(i=0;i<A->ncols;i++)
    {
        y->d[i] = ylocal[i];
    }
}

void csr_matrix_print(mat_csr *M) {
    int i;
    int t = 10;
    printf("values: ");
    for (i = 0; i < t; i++) {
        printf("%f ", M->values[i]);
    }
    printf("\ncolumns: ");
    for (i = 0; i < t; i++) {
        printf("%d ", M->cols[i]);
    }
    printf("\npointerB: ");
    for (i = 0; i < t; i++) {
        printf("%d\t", M->pointerB[i]);
    }
    printf("\npointerE: ");
    for (i = 0; i < t; i++) {
        printf("%d\t", M->pointerE[i]);
    }
    printf("\n");
}

