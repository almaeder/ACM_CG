#include "preconditioner.h"


Preconditioner_none::Preconditioner_none(
    Distributed_matrix &A_distributed
){
    rows_this_rank = A_distributed.rows_this_rank;
}
void Preconditioner_none::apply_preconditioner(
    double *z_d,
    double *r_d,
    hipStream_t &default_stream
){
    hipMemcpy(z_d, r_d, sizeof(double) * rows_this_rank, hipMemcpyDeviceToDevice);
}

Preconditioner_jacobi::Preconditioner_jacobi(
    Distributed_matrix &A_distributed
){
    rows_this_rank = A_distributed.rows_this_rank;
    cudaErrchk(hipMalloc(&diag_inv_d, rows_this_rank*sizeof(double)));
    extract_diagonal_inv(
        A_distributed.data_d[0],
        A_distributed.col_indices_d[0],
        A_distributed.row_ptr_d[0],
        diag_inv_d,
        rows_this_rank
    );
}
void Preconditioner_jacobi::apply_preconditioner(
    double *z_d,
    double *r_d,
    hipStream_t &default_stream
){
    elementwise_vector_vector(
        r_d,
        diag_inv_d,
        z_d,
        rows_this_rank,
        default_stream
    ); 
}