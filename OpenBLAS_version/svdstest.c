#include "svds.h"
#include "string.h"

int m;
int n;

void dense_test()
{
    printf("Test svds_C on dense matrix begin.\n");
    //Construct a random dense matrix
    m = 2000;
    n = 1000;
    mat* D = matrix_new(m, n);
    initialize_random_matrix_double(D);
    int k = 100;
    int i;
    struct timeval start_timeval, end_timeval;
    
   
    mat *UU;
    mat *VV;
    mat *SS;

    gettimeofday(&start_timeval, NULL);
    
    //svds_C with user's options
    svds_C_dense_opt(D, &UU, &SS, &VV, k, 1e-10, 3*k, 10);
    
    //svds_C with settled options
    //svds_C_dense(D, &UU, &SS, &VV, k);
    
    gettimeofday(&end_timeval, NULL);   
    
    mat* U = matrix_new(m ,n);
    mat* S = matrix_new(n ,n);
    mat* Vt = matrix_new(n ,n);
    
    singular_value_decomposition(D, U, S, Vt);
    
    printf("  The singular values computed by svds_C and SVD:\n");
    for(i = 0; i < k; i++)
        printf("    %.16lf %.16lf\n", SS->d[i], S->d[i*n+i]);
        
    printf("  The runtime of svds-C is: %f seconds\n", get_seconds_frac(start_timeval,end_timeval));

    matrix_delete(UU);
    matrix_delete(VV);
    matrix_delete(SS);
    matrix_delete(U);
    matrix_delete(S);
    matrix_delete(Vt);
    matrix_delete(D);
    printf("Test svds_C on dense matrix end.\n");
}

void sparse_test()
{
    FILE* fid;
    printf("Test svds_C on sparse matrix begin.\n");
    //Read matrix in coo format from file
    fid = fopen("SNAP.dat", "r");
    m = 82168;
    n = m;
    int nnz = 948464;

    mat_coo *A = coo_matrix_new(m, n, nnz);
    A->nnz = nnz;
    long long i;
    for(i=0;i<A->nnz;i++)
    {
        int ii, jj;
        double kk;
        fscanf(fid, "%d %d %lf", &ii, &jj, &kk);
        A->rows[i] = (int)ii;
        A->cols[i] = (int)jj;
        A->values[i] = kk;
    }
    mat_csr* D = csr_matrix_new();
    
    //Convert coo format to csr format
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);

    struct timeval start_timeval, end_timeval;
    
        
    int k = 100;
    
    mat *UU;
    mat *VV;
    mat *SS;
    gettimeofday(&start_timeval, NULL);
    
    //svds_C with user's options
    svds_C_opt(D, &UU, &SS, &VV, k, 1e-10, 3*k, 10);
    
    //svds_C with default options
    //svds_C(D, &UU, &SS, &VV, k);
    gettimeofday(&end_timeval, NULL);
    
    printf("  Singular values computed by svds_C:\n");
    for(i = 0; i < k; i++)
        printf("    %.16lf\n", SS->d[i]);
    
    printf("  The runtime of svds-C is: %f seconds\n", get_seconds_frac(start_timeval,end_timeval));
    
    matrix_delete(UU);
    matrix_delete(VV);
    matrix_delete(SS);
    csr_matrix_delete(D);
    printf("Test svds_C on sparse matrix end.\n");
}


int main(int argc, char const *argv[])
{
    sparse_test();
    dense_test();
    return 0;
}
