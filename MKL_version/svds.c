#include "svds.h"

void x_minus_VVTx(mat *V, vec *x, vec *xt)
{
    matrix_transpose_vector_mult(V, x, xt);
    double alpha, beta;
    alpha = -1.0;
    beta = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans, V->nrows, V->ncols, alpha, V->d, V->nrows, xt->d, 1, beta, x->d, 1);
}

void AATx(mat_csr *A, vec *x, vec *xt)
{
    csr_matrix_transpose_vector_mult(A, x, xt);
    csr_matrix_vector_mult(A, xt, x);
}

void matrix_set_colm(mat* M, long long j, vec *column_vec)
{
    int i;
    #pragma omp parallel shared(column_vec,M,j) private(i)
    {
    #pragma omp for
    for(i = 0; i < M->nrows; ++i)
        M->d[j*M->nrows+i] = column_vec->d[i];
    }
}

void vector_scale_d(vec *v, double scalar)
{
    int i;
    #pragma omp parallel shared(v,scalar) private(i) 
    {
    #pragma omp for
    for(i=0; i<(v->nrows); i++)
        v->d[i] = (v->d[i])/scalar;
    }
}

void svds_C(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k)
{
    const double eps = 1e-10;
    const int b = max(3*k, 15);
    const int maxiter = 10;
    mat *U = matrix_new(A->nrows, b);
    mat *V = matrix_new(A->ncols, b);
    mat *rm = matrix_new(A->ncols, 1);
    double alpha[b+1];
    double beta[b+1];
    initialize_random_matrix(rm);
    vec *vt = vector_new(A->ncols);
    matrix_get_col(rm, 0, vt);
    double nv = cblas_dnrm2(A->ncols, vt->d, 1);
    vector_scale_d(vt, nv);
    int i;
    vec *ut = vector_new(A->nrows);
    vec *utb = vector_new(A->nrows);
    vec *vtb = vector_new(A->ncols);
    vec *bt = vector_new(b);
    double ai, bi, ntol;
    matrix_set_colm(V, 0, vt);
    double sk_last=-1;
    mat *B_now;
    mat *UB;
    mat *SB;
    mat *VB;
    for(i = 0; i < b; ++i)
    { 	
        vector_copy(utb, ut);
        vector_copy(vtb, vt);
        csr_matrix_vector_mult(A, vt, ut);
        if (i == 0)
        {
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(vt->nrows, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        else
        {
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            ntol = vector_dot_product(ut, utb);
            {
                bt->nrows = i;
                U->ncols = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
    }
    int flag = 0;
    double gamma[k];
    B_now = matrix_new(b, b);
    int j;
    for(j=0;j<b-1;j++)
    {
        matrix_set_element(B_now, j, j, alpha[j]);
        matrix_set_element(B_now, j, j+1, beta[j]);
    }
    matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
    UB = matrix_new(b, b);
    SB = matrix_new(b, b);
    VB = matrix_new(b, b);
    singular_value_decomposition(B_now, UB, SB, VB);
    for(i=0;i<k;i++)
    {
        gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
        double gi = gamma[i];
        if(gi < 0) gi = -gi;
        flag += (gi < eps*matrix_get_element(SB, i, i));
    }
    U->ncols = b;
    V->ncols = b;
    bt->nrows = b;
    int inds[k];
    (*Sk) = matrix_new(k, 1);
    for(i=0;i<b;i++)
    {
        alpha[i] = 0;
        beta[i] = 0;
    }
    for(i=0;i<k;i++)
    {
        inds[i] = i;
        (*Sk)->d[i] = matrix_get_element(SB, i, i);
        alpha[i] = matrix_get_element(SB, i, i);
        beta[i] = 0;
    }
    mat* UBk = matrix_new(b, k);
    mat* VBt = matrix_new(b, b);
    matrix_build_transpose(VBt, VB);
    mat* VBk = matrix_new(b, k);
    matrix_get_selected_columns(UB, inds, UBk);
    matrix_get_selected_columns(VBt, inds, VBk);
    (*Uk) = matrix_new(A->nrows, k);
    matrix_matrix_mult(U, UBk, *Uk);
    (*Vk) = matrix_new(A->ncols, k);
    matrix_matrix_mult(V, VBk, *Vk);
    int iters = 1;
    
    while(iters < maxiter)
    {
        if(flag==k)
            break;
        V->ncols = k;
        U->ncols = k;
        matrix_copy(V, *Vk);
        matrix_copy(U, *Uk);
        nv = 0;
        nv = cblas_dnrm2(A->ncols, vt->d, 1);
        vector_scale_d(vt, nv);
        V->ncols++;
        matrix_set_colm(V, k, vt);

        csr_matrix_vector_mult(A, vt, ut);
        bt->nrows = k;
        x_minus_VVTx(U, ut, bt);
        U->ncols++;
        ai = cblas_dnrm2(ut->nrows, ut->d, 1);
        alpha[k] = ai;
        vector_scale_d(ut, ai);
        matrix_set_colm(U, k, ut);

        vector_copy(vtb, vt);
        csr_matrix_transpose_vector_mult(A, ut, vt);
        cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
        bt->nrows = k+1;
        x_minus_VVTx(V, vt, bt);
        V->ncols++;
        bi = cblas_dnrm2(vt->nrows, vt->d, 1);
        beta[k] = bi;
        vector_scale_d(vt, bi);
        matrix_set_colm(V, k+1, vt);
        for(i = k+1; i < b; ++i)
        { 	
            vector_copy(utb, ut);
            vector_copy(vtb, vt);
            csr_matrix_vector_mult(A, vt, ut);
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            {
                bt->nrows = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            {
                bt->nrows = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        matrix_delete(B_now);
        B_now = matrix_new(b, b);
        for(j=0;j<k;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, k, gamma[j]);
        }
        for(j=k;j<b-1;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, j+1, beta[j]);
        }
        matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
        singular_value_decomposition(B_now, UB, SB, VB);
        flag = 0;
        for(i=0;i<k;i++)
        {
            gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
            double gi = gamma[i];
            if(gi < 0) gi = -gi;
            flag += (gi < eps * matrix_get_element(SB, i, i));
        }
        U->ncols = b;
        V->ncols = b;
        bt->nrows = b;
        for(i=0;i<k;i++)
        {
            inds[i] = i;
            (*Sk)->d[i] = matrix_get_element(SB, i, i);
            alpha[i] = matrix_get_element(SB, i, i);
            beta[i] = 0;
        }
        matrix_build_transpose(VBt, VB);
        matrix_get_selected_columns(UB, inds, UBk);
        matrix_get_selected_columns(VBt, inds, VBk);
        matrix_matrix_mult(U, UBk, *Uk);
        matrix_matrix_mult(V, VBk, *Vk);
        matrix_transpose_matrix_mult(V, V, B_now);
        iters++;
    }
    
    matrix_delete(B_now);
    matrix_delete(SB);
    matrix_delete(VB);
    matrix_delete(UB);
    matrix_delete(VBt);
    matrix_delete(U);
    matrix_delete(UBk);
    matrix_delete(V);
    matrix_delete(VBk);
    matrix_delete(rm);
    vector_delete(bt);
    vector_delete(ut);
    vector_delete(utb);
    vector_delete(vt);
    vector_delete(vtb);
}

void svds_C_opt(mat_csr *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter)
{
    const int b = maxbasis;
    mat *U = matrix_new(A->nrows, b);
    mat *V = matrix_new(A->ncols, b);
    mat *rm = matrix_new(A->ncols, 1);
    double alpha[b+1];
    double beta[b+1];
    initialize_random_matrix(rm);
    vec *vt = vector_new(A->ncols);
    matrix_get_col(rm, 0, vt);
    double nv = cblas_dnrm2(A->ncols, vt->d, 1);
    vector_scale_d(vt, nv);
    int i;
    vec *ut = vector_new(A->nrows);
    vec *utb = vector_new(A->nrows);
    vec *vtb = vector_new(A->ncols);
    vec *bt = vector_new(b);
    double ai, bi, ntol;
    matrix_set_colm(V, 0, vt);
    double sk_last=-1;
    mat *B_now;
    mat *UB;
    mat *SB;
    mat *VB;
    for(i = 0; i < b; ++i)
    { 	
        vector_copy(utb, ut);
        vector_copy(vtb, vt);
        csr_matrix_vector_mult(A, vt, ut);
        if (i == 0)
        {
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(vt->nrows, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        else
        {
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            ntol = vector_dot_product(ut, utb);
            {
                bt->nrows = i;
                U->ncols = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
    }
    int flag = 0;
    double gamma[k];
    B_now = matrix_new(b, b);
    int j;
    for(j=0;j<b-1;j++)
    {
        matrix_set_element(B_now, j, j, alpha[j]);
        matrix_set_element(B_now, j, j+1, beta[j]);
    }
    matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
    UB = matrix_new(b, b);
    SB = matrix_new(b, b);
    VB = matrix_new(b, b);
    singular_value_decomposition(B_now, UB, SB, VB);
    for(i=0;i<k;i++)
    {
        gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
        double gi = gamma[i];
        if(gi < 0) gi = -gi;
        flag += (gi < eps*matrix_get_element(SB, i, i));
    }
    U->ncols = b;
    V->ncols = b;
    bt->nrows = b;
    int inds[k];
    (*Sk) = matrix_new(k, 1);
    for(i=0;i<b;i++)
    {
        alpha[i] = 0;
        beta[i] = 0;
    }
    for(i=0;i<k;i++)
    {
        inds[i] = i;
        (*Sk)->d[i] = matrix_get_element(SB, i, i);
        alpha[i] = matrix_get_element(SB, i, i);
        beta[i] = 0;
    }
    mat* UBk = matrix_new(b, k);
    mat* VBt = matrix_new(b, b);
    matrix_build_transpose(VBt, VB);
    mat* VBk = matrix_new(b, k);
    matrix_get_selected_columns(UB, inds, UBk);
    matrix_get_selected_columns(VBt, inds, VBk);
    (*Uk) = matrix_new(A->nrows, k);
    matrix_matrix_mult(U, UBk, *Uk);
    (*Vk) = matrix_new(A->ncols, k);
    matrix_matrix_mult(V, VBk, *Vk);
    int iters = 1;
    
    while(iters < maxiter)
    {
        if(flag==k)
            break;
        V->ncols = k;
        U->ncols = k;
        matrix_copy(V, *Vk);
        matrix_copy(U, *Uk);
        nv = 0;
        nv = cblas_dnrm2(A->ncols, vt->d, 1);
        vector_scale_d(vt, nv);
        V->ncols++;
        matrix_set_colm(V, k, vt);

        csr_matrix_vector_mult(A, vt, ut);
        bt->nrows = k;
        x_minus_VVTx(U, ut, bt);
        U->ncols++;
        ai = cblas_dnrm2(ut->nrows, ut->d, 1);
        alpha[k] = ai;
        vector_scale_d(ut, ai);
        matrix_set_colm(U, k, ut);

        vector_copy(vtb, vt);
        csr_matrix_transpose_vector_mult(A, ut, vt);
        cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
        bt->nrows = k+1;
        x_minus_VVTx(V, vt, bt);
        V->ncols++;
        bi = cblas_dnrm2(vt->nrows, vt->d, 1);
        beta[k] = bi;
        vector_scale_d(vt, bi);
        matrix_set_colm(V, k+1, vt);
        for(i = k+1; i < b; ++i)
        { 	
            vector_copy(utb, ut);
            vector_copy(vtb, vt);
            csr_matrix_vector_mult(A, vt, ut);
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            {
                bt->nrows = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            csr_matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            {
                bt->nrows = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        matrix_delete(B_now);
        B_now = matrix_new(b, b);
        for(j=0;j<k;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, k, gamma[j]);
        }
        for(j=k;j<b-1;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, j+1, beta[j]);
        }
        matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
        singular_value_decomposition(B_now, UB, SB, VB);
        flag = 0;
        for(i=0;i<k;i++)
        {
            gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
            double gi = gamma[i];
            if(gi < 0) gi = -gi;
            flag += (gi < eps * matrix_get_element(SB, i, i));
        }
        U->ncols = b;
        V->ncols = b;
        bt->nrows = b;
        for(i=0;i<k;i++)
        {
            inds[i] = i;
            (*Sk)->d[i] = matrix_get_element(SB, i, i);
            alpha[i] = matrix_get_element(SB, i, i);
            beta[i] = 0;
        }
        matrix_build_transpose(VBt, VB);
        matrix_get_selected_columns(UB, inds, UBk);
        matrix_get_selected_columns(VBt, inds, VBk);
        matrix_matrix_mult(U, UBk, *Uk);
        matrix_matrix_mult(V, VBk, *Vk);
        matrix_transpose_matrix_mult(V, V, B_now);
        iters++;
    }
    
    matrix_delete(B_now);
    matrix_delete(SB);
    matrix_delete(VB);
    matrix_delete(UB);
    matrix_delete(VBt);
    matrix_delete(U);
    matrix_delete(UBk);
    matrix_delete(V);
    matrix_delete(VBk);
    matrix_delete(rm);
    vector_delete(bt);
    vector_delete(ut);
    vector_delete(utb);
    vector_delete(vt);
    vector_delete(vtb);
}

void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k)
{
    const double eps = 1e-10;
    const int b = max(3*k, 15);
    const int maxiter = 10;
    mat *U = matrix_new(A->nrows, b);
    mat *V = matrix_new(A->ncols, b);
    mat *rm = matrix_new(A->ncols, 1);
    double alpha[b+1];
    double beta[b+1];
    initialize_random_matrix(rm);
    vec *vt = vector_new(A->ncols);
    matrix_get_col(rm, 0, vt);
    double nv = cblas_dnrm2(A->ncols, vt->d, 1);
    vector_scale_d(vt, nv);
    int i;
    vec *ut = vector_new(A->nrows);
    vec *utb = vector_new(A->nrows);
    vec *vtb = vector_new(A->ncols);
    vec *bt = vector_new(b);
    double ai, bi, ntol;
    matrix_set_colm(V, 0, vt);
    double sk_last=-1;
    mat *B_now;
    mat *UB;
    mat *SB;
    mat *VB;
    for(i = 0; i < b; ++i)
    { 	
        vector_copy(utb, ut);
        vector_copy(vtb, vt);
        matrix_vector_mult(A, vt, ut);
        if (i == 0)
        {
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(vt->nrows, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        else
        {
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            ntol = vector_dot_product(ut, utb);
            {
                bt->nrows = i;
                U->ncols = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
    }
    int flag = 0;
    double gamma[k];
    B_now = matrix_new(b, b);
    int j;
    for(j=0;j<b-1;j++)
    {
        matrix_set_element(B_now, j, j, alpha[j]);
        matrix_set_element(B_now, j, j+1, beta[j]);
    }
    matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
    UB = matrix_new(b, b);
    SB = matrix_new(b, b);
    VB = matrix_new(b, b);
    singular_value_decomposition(B_now, UB, SB, VB);
    for(i=0;i<k;i++)
    {
        gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
        double gi = gamma[i];
        if(gi < 0) gi = -gi;
        flag += (gi < eps*matrix_get_element(SB, i, i));
    }
    U->ncols = b;
    V->ncols = b;
    bt->nrows = b;
    int inds[k];
    (*Sk) = matrix_new(k, 1);
    for(i=0;i<b;i++)
    {
        alpha[i] = 0;
        beta[i] = 0;
    }
    for(i=0;i<k;i++)
    {
        inds[i] = i;
        (*Sk)->d[i] = matrix_get_element(SB, i, i);
        alpha[i] = matrix_get_element(SB, i, i);
        beta[i] = 0;
    }
    mat* UBk = matrix_new(b, k);
    mat* VBt = matrix_new(b, b);
    matrix_build_transpose(VBt, VB);
    mat* VBk = matrix_new(b, k);
    matrix_get_selected_columns(UB, inds, UBk);
    matrix_get_selected_columns(VBt, inds, VBk);
    (*Uk) = matrix_new(A->nrows, k);
    matrix_matrix_mult(U, UBk, *Uk);
    (*Vk) = matrix_new(A->ncols, k);
    matrix_matrix_mult(V, VBk, *Vk);
    int iters = 1;
    
    while(iters < maxiter)
    {
        if(flag==k)
            break;
        V->ncols = k;
        U->ncols = k;
        matrix_copy(V, *Vk);
        matrix_copy(U, *Uk);
        nv = 0;
        nv = cblas_dnrm2(A->ncols, vt->d, 1);
        vector_scale_d(vt, nv);
        V->ncols++;
        matrix_set_colm(V, k, vt);

        matrix_vector_mult(A, vt, ut);
        bt->nrows = k;
        x_minus_VVTx(U, ut, bt);
        U->ncols++;
        ai = cblas_dnrm2(ut->nrows, ut->d, 1);
        alpha[k] = ai;
        vector_scale_d(ut, ai);
        matrix_set_colm(U, k, ut);

        vector_copy(vtb, vt);
        matrix_transpose_vector_mult(A, ut, vt);
        cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
        bt->nrows = k+1;
        x_minus_VVTx(V, vt, bt);
        V->ncols++;
        bi = cblas_dnrm2(vt->nrows, vt->d, 1);
        beta[k] = bi;
        vector_scale_d(vt, bi);
        matrix_set_colm(V, k+1, vt);
        for(i = k+1; i < b; ++i)
        { 	
            vector_copy(utb, ut);
            vector_copy(vtb, vt);
            matrix_vector_mult(A, vt, ut);
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            {
                bt->nrows = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            {
                bt->nrows = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        matrix_delete(B_now);
        B_now = matrix_new(b, b);
        for(j=0;j<k;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, k, gamma[j]);
        }
        for(j=k;j<b-1;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, j+1, beta[j]);
        }
        matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
        singular_value_decomposition(B_now, UB, SB, VB);
        flag = 0;
        for(i=0;i<k;i++)
        {
            gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
            double gi = gamma[i];
            if(gi < 0) gi = -gi;
            flag += (gi < eps * matrix_get_element(SB, i, i));
        }
        U->ncols = b;
        V->ncols = b;
        bt->nrows = b;
        for(i=0;i<k;i++)
        {
            inds[i] = i;
            (*Sk)->d[i] = matrix_get_element(SB, i, i);
            alpha[i] = matrix_get_element(SB, i, i);
            beta[i] = 0;
        }
        matrix_build_transpose(VBt, VB);
        matrix_get_selected_columns(UB, inds, UBk);
        matrix_get_selected_columns(VBt, inds, VBk);
        matrix_matrix_mult(U, UBk, *Uk);
        matrix_matrix_mult(V, VBk, *Vk);
        matrix_transpose_matrix_mult(V, V, B_now);
        iters++;
    }
    matrix_delete(B_now);
    matrix_delete(SB);
    matrix_delete(VB);
    matrix_delete(UB);
    matrix_delete(VBt);
    matrix_delete(U);
    matrix_delete(UBk);
    matrix_delete(V);
    matrix_delete(VBk);
    matrix_delete(rm);
    vector_delete(bt);
    vector_delete(ut);
    vector_delete(utb);
    vector_delete(vt);
    vector_delete(vtb);
}

void svds_C_dense_opt(mat *A, mat **Uk, mat **Sk, mat **Vk, int k, double eps, int maxbasis, int maxiter)
{
    const int b = maxbasis;
    mat *U = matrix_new(A->nrows, b);
    mat *V = matrix_new(A->ncols, b);
    mat *rm = matrix_new(A->ncols, 1);
    double alpha[b+1];
    double beta[b+1];
    initialize_random_matrix(rm);
    vec *vt = vector_new(A->ncols);
    matrix_get_col(rm, 0, vt);
    double nv = cblas_dnrm2(A->ncols, vt->d, 1);
    vector_scale_d(vt, nv);
    int i;
    vec *ut = vector_new(A->nrows);
    vec *utb = vector_new(A->nrows);
    vec *vtb = vector_new(A->ncols);
    vec *bt = vector_new(b);
    double ai, bi, ntol;
    matrix_set_colm(V, 0, vt);
    double sk_last=-1;
    mat *B_now;
    mat *UB;
    mat *SB;
    mat *VB;
    for(i = 0; i < b; ++i)
    { 	
        vector_copy(utb, ut);
        vector_copy(vtb, vt);
        matrix_vector_mult(A, vt, ut);
        if (i == 0)
        {
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(vt->nrows, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        else
        {
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            ntol = vector_dot_product(ut, utb);
            {
                bt->nrows = i;
                U->ncols = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            ntol = vector_dot_product(vt, vtb);
            {
                bt->nrows = i+1;
                V->ncols = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
    }
    int flag = 0;
    double gamma[k];
    B_now = matrix_new(b, b);
    int j;
    for(j=0;j<b-1;j++)
    {
        matrix_set_element(B_now, j, j, alpha[j]);
        matrix_set_element(B_now, j, j+1, beta[j]);
    }
    matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
    UB = matrix_new(b, b);
    SB = matrix_new(b, b);
    VB = matrix_new(b, b);
    singular_value_decomposition(B_now, UB, SB, VB);
    for(i=0;i<k;i++)
    {
        gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
        double gi = gamma[i];
        if(gi < 0) gi = -gi;
        flag += (gi < eps*matrix_get_element(SB, i, i));
    }
    U->ncols = b;
    V->ncols = b;
    bt->nrows = b;
    int inds[k];
    (*Sk) = matrix_new(k, 1);
    for(i=0;i<b;i++)
    {
        alpha[i] = 0;
        beta[i] = 0;
    }
    for(i=0;i<k;i++)
    {
        inds[i] = i;
        (*Sk)->d[i] = matrix_get_element(SB, i, i);
        alpha[i] = matrix_get_element(SB, i, i);
        beta[i] = 0;
    }
    mat* UBk = matrix_new(b, k);
    mat* VBt = matrix_new(b, b);
    matrix_build_transpose(VBt, VB);
    mat* VBk = matrix_new(b, k);
    matrix_get_selected_columns(UB, inds, UBk);
    matrix_get_selected_columns(VBt, inds, VBk);
    (*Uk) = matrix_new(A->nrows, k);
    matrix_matrix_mult(U, UBk, *Uk);
    (*Vk) = matrix_new(A->ncols, k);
    matrix_matrix_mult(V, VBk, *Vk);
    int iters = 1;
    
    while(iters < maxiter)
    {
        if(flag==k)
            break;
        V->ncols = k;
        U->ncols = k;
        matrix_copy(V, *Vk);
        matrix_copy(U, *Uk);
        nv = 0;
        nv = cblas_dnrm2(A->ncols, vt->d, 1);
        vector_scale_d(vt, nv);
        V->ncols++;
        matrix_set_colm(V, k, vt);

        matrix_vector_mult(A, vt, ut);
        bt->nrows = k;
        x_minus_VVTx(U, ut, bt);
        U->ncols++;
        ai = cblas_dnrm2(ut->nrows, ut->d, 1);
        alpha[k] = ai;
        vector_scale_d(ut, ai);
        matrix_set_colm(U, k, ut);

        vector_copy(vtb, vt);
        matrix_transpose_vector_mult(A, ut, vt);
        cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
        bt->nrows = k+1;
        x_minus_VVTx(V, vt, bt);
        V->ncols++;
        bi = cblas_dnrm2(vt->nrows, vt->d, 1);
        beta[k] = bi;
        vector_scale_d(vt, bi);
        matrix_set_colm(V, k+1, vt);
        for(i = k+1; i < b; ++i)
        { 	
            vector_copy(utb, ut);
            vector_copy(vtb, vt);
            matrix_vector_mult(A, vt, ut);
            cblas_daxpy(A->nrows, -bi, utb->d, 1, ut->d, 1);
            {
                bt->nrows = i;
                x_minus_VVTx(U, ut, bt);
            }
            U->ncols++;
            ai = cblas_dnrm2(ut->nrows, ut->d, 1);
            alpha[i] = ai;
            vector_scale_d(ut, ai);
            matrix_set_colm(U, i, ut);
            matrix_transpose_vector_mult(A, ut, vt);
            cblas_daxpy(A->ncols, -ai, vtb->d, 1, vt->d, 1);
            {
                bt->nrows = i+1;
                x_minus_VVTx(V, vt, bt);
            }
            V->ncols++;
            bi = cblas_dnrm2(vt->nrows, vt->d, 1);
            beta[i] = bi;
            if (i==b-1)
                break;
            vector_scale_d(vt, bi);
            matrix_set_colm(V, i+1, vt);
        }
        matrix_delete(B_now);
        B_now = matrix_new(b, b);
        for(j=0;j<k;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, k, gamma[j]);
        }
        for(j=k;j<b-1;j++)
        {
            matrix_set_element(B_now, j, j, alpha[j]);
            matrix_set_element(B_now, j, j+1, beta[j]);
        }
        matrix_set_element(B_now, b-1, b-1, alpha[b-1]);
        singular_value_decomposition(B_now, UB, SB, VB);
        flag = 0;
        for(i=0;i<k;i++)
        {
            gamma[i] = beta[b-1]*matrix_get_element(UB, b-1, i);
            double gi = gamma[i];
            if(gi < 0) gi = -gi;
            flag += (gi < eps * matrix_get_element(SB, i, i));
        }
        U->ncols = b;
        V->ncols = b;
        bt->nrows = b;
        for(i=0;i<k;i++)
        {
            inds[i] = i;
            (*Sk)->d[i] = matrix_get_element(SB, i, i);
            alpha[i] = matrix_get_element(SB, i, i);
            beta[i] = 0;
        }
        matrix_build_transpose(VBt, VB);
        matrix_get_selected_columns(UB, inds, UBk);
        matrix_get_selected_columns(VBt, inds, VBk);
        matrix_matrix_mult(U, UBk, *Uk);
        matrix_matrix_mult(V, VBk, *Vk);
        matrix_transpose_matrix_mult(V, V, B_now);
        iters++;
        //puts("hahah");
    }
    //puts("121212");
    matrix_delete(B_now);
    matrix_delete(SB);
    matrix_delete(VB);
    matrix_delete(UB);
    matrix_delete(VBt);
    matrix_delete(U);
    matrix_delete(UBk);
    matrix_delete(V);
    matrix_delete(VBk);
    matrix_delete(rm);
    vector_delete(bt);
    vector_delete(ut);
    vector_delete(utb);
    vector_delete(vt);
    vector_delete(vtb);
}
