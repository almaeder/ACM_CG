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
        A_distributed.datas_d[0],
        A_distributed.col_inds_d[0],
        A_distributed.row_ptrs_d[0],
        diag_inv_d,
        rows_this_rank
    );
}
Preconditioner_jacobi::~Preconditioner_jacobi(){
    cudaErrchk(
        hipFree(diag_inv_d)
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

Preconditioner_jacobi_split::Preconditioner_jacobi_split(
    Distributed_matrix &A_distributed,
    Distributed_subblock &A_subblock_distributed
){
    rows_this_rank = A_distributed.rows_this_rank;
    cudaErrchk(hipMalloc(&diag_inv_d, rows_this_rank*sizeof(double)));
    cudaErrchk(hipMemset(diag_inv_d, 0, rows_this_rank*sizeof(double)))
    extract_diagonal(
        A_distributed.datas_d[0],
        A_distributed.col_inds_d[0],
        A_distributed.row_ptrs_d[0],
        diag_inv_d,
        rows_this_rank
    );
    extract_add_subblock_diagonal(
        A_subblock_distributed.subblock_indices_local_d,
        A_subblock_distributed.row_ptr_compressed_d,
        A_subblock_distributed.col_indices_compressed_d,
        A_subblock_distributed.data_d,
        diag_inv_d,
        A_subblock_distributed.counts_subblock[A_subblock_distributed.rank],
        A_subblock_distributed.displacements_subblock[A_subblock_distributed.rank]
    );
    inv_inplace(
        diag_inv_d,
        rows_this_rank
    );

}
Preconditioner_jacobi_split::~Preconditioner_jacobi_split(){
    cudaErrchk(
        hipFree(diag_inv_d)
    );
}

void Preconditioner_jacobi_split::apply_preconditioner(
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