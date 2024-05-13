#include "preconditioner.h"


Preconditioner_none::Preconditioner_none(
    Distributed_matrix &A_distributed
){
    rows_this_rank = A_distributed.rows_this_rank;
}
void Preconditioner_none::apply_preconditioner(
    double *z_d,
    double *r_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
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
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
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
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
){
    elementwise_vector_vector(
        r_d,
        diag_inv_d,
        z_d,
        rows_this_rank,
        default_stream
    ); 
}

Preconditioner_block_icholesky::Preconditioner_block_icholesky(
    Distributed_matrix &A_distributed
){
    rows_this_rank = A_distributed.rows_this_rank;
    nnz = A_distributed.nnz_per_neighbour[0];

    cudaErrchk(hipMalloc(&row_ptr_d, (rows_this_rank+1)*sizeof(int)));
    cudaErrchk(hipMalloc(&col_ind_d, nnz*sizeof(int)));
    cudaErrchk(hipMalloc(&data_d, nnz*sizeof(double)));
    cudaErrchk(hipMalloc(&y_d, rows_this_rank*sizeof(double)));
    cudaErrchk(hipMalloc(&x_d, rows_this_rank*sizeof(double)));

    hipMemcpy(row_ptr_d, A_distributed.row_ptrs_d[0], (rows_this_rank+1)*sizeof(int), hipMemcpyDeviceToDevice);
    hipMemcpy(col_ind_d, A_distributed.col_inds_d[0], nnz*sizeof(int), hipMemcpyDeviceToDevice);
    hipMemcpy(data_d, A_distributed.datas_d[0], nnz*sizeof(double), hipMemcpyDeviceToDevice);

    // Create matrix descriptor for M
    rocsparse_mat_descr descr_M;
    rocsparse_create_mat_descr(&descr_M);

    // Create matrix descriptor for L
    rocsparse_create_mat_descr(&descr_L);
    rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower);
    rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_unit);

    // Create matrix descriptor for L'
    rocsparse_create_mat_descr(&descr_Lt);
    rocsparse_set_mat_fill_mode(descr_Lt, rocsparse_fill_mode_upper);
    rocsparse_set_mat_diag_type(descr_Lt, rocsparse_diag_type_non_unit);

    // Create matrix info structure
    rocsparse_create_mat_info(&info);
    // Obtain required buffer size
    size_t buffer_size_M;
    size_t buffer_size_L;
    size_t buffer_size_Lt;
    rocsparse_dcsric0_buffer_size(A_distributed.default_rocsparseHandle,
                                rows_this_rank,
                                nnz,
                                descr_M,
                                data_d,
                                row_ptr_d,
                                col_ind_d,
                                info,
                                &buffer_size_M);
    rocsparse_dcsrsv_buffer_size(A_distributed.default_rocsparseHandle,
                                rocsparse_operation_none,
                                rows_this_rank,
                                nnz,
                                descr_L,
                                data_d,
                                row_ptr_d,
                                col_ind_d,
                                info,
                                &buffer_size_L);
    rocsparse_dcsrsv_buffer_size(A_distributed.default_rocsparseHandle,
                                rocsparse_operation_transpose,
                                rows_this_rank,
                                nnz,
                                descr_Lt,
                                data_d,
                                row_ptr_d,
                                col_ind_d,
                                info,
                                &buffer_size_Lt);
    size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));

    // Allocate temporary buffer
    cudaErrchk(hipMalloc(&temp_buffer_d, buffer_size));

    // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
    // computation performance
    rocsparse_dcsric0_analysis(A_distributed.default_rocsparseHandle,
                            rows_this_rank,
                            nnz,
                            descr_M,
                            data_d,
                            row_ptr_d,
                            col_ind_d,
                            info,
                            rocsparse_analysis_policy_reuse,
                            rocsparse_solve_policy_auto,
                            temp_buffer_d);
    rocsparse_dcsrsv_analysis(A_distributed.default_rocsparseHandle,
                            rocsparse_operation_none,
                            rows_this_rank,
                            nnz,
                            descr_L,
                            data_d,
                            row_ptr_d,
                            col_ind_d,
                            info,
                            rocsparse_analysis_policy_reuse,
                            rocsparse_solve_policy_auto,
                            temp_buffer_d);
    rocsparse_dcsrsv_analysis(A_distributed.default_rocsparseHandle,
                            rocsparse_operation_transpose,
                            rows_this_rank,
                            nnz,
                            descr_Lt,
                            data_d,
                            row_ptr_d,
                            col_ind_d,
                            info,
                            rocsparse_analysis_policy_reuse,
                            rocsparse_solve_policy_auto,
                            temp_buffer_d);
    // Check for zero pivot
    rocsparse_int position;
    if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(A_distributed.default_rocsparseHandle,
                                                                info,
                                                                &position))
    {
        printf("A has structural zero at A(%d,%d)\n", position, position);
        exit(1);
    }

    // Compute incomplete Cholesky factorization M = LL'
    rocsparse_dcsric0(A_distributed.default_rocsparseHandle,
                    rows_this_rank,
                    nnz,
                    descr_M,
                    data_d,
                    row_ptr_d,
                    col_ind_d,
                    info,
                    rocsparse_solve_policy_auto,
                    temp_buffer_d);

    // Check for zero pivot
    if(rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(A_distributed.default_rocsparseHandle,
                                                                info,
                                                                &position))
    {
        printf("L has structural and/or numerical zero at L(%d,%d)\n",
            position,
            position);
        exit(1);
    }

    rocsparse_destroy_mat_descr(descr_M);

}
Preconditioner_block_icholesky::~Preconditioner_block_icholesky(){
    
    cudaErrchk(hipFree(row_ptr_d));
    cudaErrchk(hipFree(col_ind_d));
    cudaErrchk(hipFree(data_d));
    cudaErrchk(hipFree(y_d));
    cudaErrchk(hipFree(temp_buffer_d));
    rocsparse_destroy_mat_info(info);
    rocsparse_destroy_mat_descr(descr_L);
    rocsparse_destroy_mat_descr(descr_Lt);    
}

void Preconditioner_block_icholesky::apply_preconditioner(
    double *z_d,
    double *r_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
){

    cudaErrchk(hipMemcpy(x_d, r_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToDevice))


    double alpha = 1.0;
    // Solve Ly = r
    rocsparse_status status = rocsparse_dcsrsv_solve(default_rocsparseHandle,
                        rocsparse_operation_none,
                        rows_this_rank,
                        nnz,
                        &alpha,
                        descr_L,
                        data_d,
                        row_ptr_d,
                        col_ind_d,
                        info,
                        x_d,
                        y_d,
                        rocsparse_solve_policy_auto,
                        temp_buffer_d);

    // Solve L'z = y
    rocsparse_status status_T = rocsparse_dcsrsv_solve(default_rocsparseHandle,
                        rocsparse_operation_transpose,
                        rows_this_rank,
                        nnz,
                        &alpha,
                        descr_Lt,
                        data_d,
                        row_ptr_d,
                        col_ind_d,
                        info,
                        y_d,
                        z_d,
                        rocsparse_solve_policy_auto,
                        temp_buffer_d);

    double *z_h = new double[rows_this_rank];
    cudaErrchk(hipMemcpy(z_h, z_d, rows_this_rank*sizeof(double), hipMemcpyDeviceToHost));
    double *r_h = new double[rows_this_rank];
    cudaErrchk(hipMemcpy(r_h, r_d, rows_this_rank*sizeof(double), hipMemcpyDeviceToHost));

    double tmp = 0;
    for(int i = 0; i < rows_this_rank; i++){
        tmp += z_h[i] * r_h[i];
    }
    std::cout << tmp << std::endl;


    // for(int i = 0; i < 10; i++){
    //     std::cout << z_h[i] << " ";
    // }
    // std::cout << std::endl;
    // for(int i = 0; i < 10; i++){
    //     std::cout << r_h[i] << " ";
    // }
    // std::cout << std::endl;

    delete[] z_h; 
    delete[] r_h;

    // exit(0);
}