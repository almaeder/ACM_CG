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
    hipsparseHandle_t &default_rocsparseHandle
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
    hipsparseHandle_t &default_rocsparseHandle
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
    hipsparseHandle_t &default_rocsparseHandle
){
    elementwise_vector_vector(
        r_d,
        diag_inv_d,
        z_d,
        rows_this_rank,
        default_stream
    ); 
}

Preconditioner_block_ilu::Preconditioner_block_ilu(
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
    hipsparseMatDescr_t descr_LLt;
    cusparseErrchk(hipsparseCreateMatDescr(
        &descr_LLt));
    cusparseErrchk(hipsparseSetMatType(
        descr_LLt, HIPSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(hipsparseSetMatIndexBase(
        descr_LLt, HIPSPARSE_INDEX_BASE_ZERO));


    hipsparseFillMode_t   fill_lower    = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t   diag_unit     = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseFillMode_t   fill_upper    = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t   diag_non_unit = HIPSPARSE_DIAG_TYPE_NON_UNIT;

    // Create matrix descriptor for L
    cusparseErrchk(hipsparseCreateCsr(
        &descr_L,
        rows_this_rank,
        rows_this_rank,
        nnz,
        row_ptr_d,
        col_ind_d,
        data_d,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO,
        HIP_R_64F
    ));
    cusparseErrchk(hipsparseSpMatSetAttribute(
        descr_L, HIPSPARSE_SPMAT_FILL_MODE,
        &fill_lower, sizeof(fill_lower)) );
    cusparseErrchk(hipsparseSpMatSetAttribute(
        descr_L, HIPSPARSE_SPMAT_DIAG_TYPE,
        &diag_unit, sizeof(diag_unit)) );

    // Create matrix descriptor for L'
    cusparseErrchk(hipsparseCreateCsr(
        &descr_Lt,
        rows_this_rank,
        rows_this_rank,
        nnz,
        row_ptr_d,
        col_ind_d,
        data_d,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO,
        HIP_R_64F
    ));
    cusparseErrchk(hipsparseSpMatSetAttribute(
        descr_Lt, HIPSPARSE_SPMAT_FILL_MODE,
        &fill_upper, sizeof(fill_upper)) );
    cusparseErrchk(hipsparseSpMatSetAttribute(
        descr_Lt, HIPSPARSE_SPMAT_DIAG_TYPE,
        &diag_non_unit, sizeof(diag_non_unit)) );


    cusparseErrchk(hipsparseCreateCsrilu02Info(
        &info
    ));

    // Obtain required buffer size
    int buffer_size_LLt;
    size_t buffer_size_L;
    size_t buffer_size_Lt;

    cusparseErrchk(hipsparseDcsrilu02_bufferSize(
        A_distributed.default_cusparseHandle,
        rows_this_rank,
        nnz,
        descr_LLt,
        data_d,
        row_ptr_d,
        col_ind_d,
        info,
        &buffer_size_LLt
    ));
    cudaErrchk(hipMalloc(&buffer_LLt_d, buffer_size_LLt));

    cusparseErrchk(hipsparseCreateDnVec(
        &vecY, rows_this_rank, y_d, HIP_R_64F
    ));
    cusparseErrchk(hipsparseCreateDnVec(
        &vecX, rows_this_rank, x_d, HIP_R_64F
    ));

    const double one = 1.0;
    cusparseErrchk( hipsparseSpSV_createDescr(&spsvDescr_L) );
    cusparseErrchk( hipsparseSpSV_bufferSize(
        A_distributed.default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &one, descr_L, vecY, vecX, HIP_R_64F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr_L, &buffer_size_L));
    cudaErrchk( hipMalloc(&buffer_L_d, buffer_size_L) );

    cusparseErrchk( hipsparseSpSV_createDescr(&spsvDescr_Lt) );
    cusparseErrchk( hipsparseSpSV_bufferSize(
        A_distributed.default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &one, descr_Lt, vecY, vecX, HIP_R_64F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr_Lt, &buffer_size_Lt));
    cudaErrchk( hipMalloc(&buffer_Lt_d, buffer_size_Lt) );


    // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
    // computation performance
    cusparseErrchk( hipsparseDcsrilu02_analysis(
        A_distributed.default_cusparseHandle,
        rows_this_rank,
        nnz,
        descr_LLt,
        data_d,
        row_ptr_d,
        col_ind_d,
        info,
        HIPSPARSE_SOLVE_POLICY_NO_LEVEL,
        buffer_LLt_d
    ));
    // Check for zero pivot
    int position;
    if(HIPSPARSE_STATUS_ZERO_PIVOT == hipsparseXcsrilu02_zeroPivot(A_distributed.default_cusparseHandle,
                                                                info,
                                                                &position))
    {
        printf("A has structural zero at A(%d,%d)\n", position, position);
        exit(1);
    }

    cusparseErrchk( hipsparseDcsrilu02(
        A_distributed.default_cusparseHandle,
        rows_this_rank,
        nnz,
        descr_LLt,
        data_d,
        row_ptr_d,
        col_ind_d,
        info,
        HIPSPARSE_SOLVE_POLICY_NO_LEVEL,
        buffer_LLt_d
    ));

    // Check for zero pivot
    if(HIPSPARSE_STATUS_ZERO_PIVOT == hipsparseXcsrilu02_zeroPivot(A_distributed.default_cusparseHandle,
                                                                info,
                                                                &position))
    {
        printf("A has structural zero at A(%d,%d)\n", position, position);
        exit(1);
    }

    cusparseErrchk(hipsparseSpSV_analysis(
        A_distributed.default_cusparseHandle, 
        HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
        descr_L, vecY, vecX, HIP_R_64F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr_L, buffer_L_d));

    cusparseErrchk(hipsparseSpSV_analysis(
        A_distributed.default_cusparseHandle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE, &one,
        descr_Lt, vecY, vecX, HIP_R_64F,
        HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr_Lt, buffer_Lt_d));


    cusparseErrchk(hipsparseDestroyMatDescr(descr_LLt));

}
Preconditioner_block_ilu::~Preconditioner_block_ilu(){
    
    cudaErrchk(hipFree(row_ptr_d));
    cudaErrchk(hipFree(col_ind_d));
    cudaErrchk(hipFree(data_d));
    cudaErrchk(hipFree(y_d));
    cudaErrchk(hipFree(x_d));
    cudaErrchk(hipFree(buffer_LLt_d));
    cudaErrchk(hipFree(buffer_L_d));
    cudaErrchk(hipFree(buffer_Lt_d));

    cusparseErrchk(hipsparseDestroyCsrilu02Info(info));
    cusparseErrchk(hipsparseDestroySpMat(descr_L));
    cusparseErrchk(hipsparseDestroySpMat(descr_Lt));    
    cusparseErrchk(hipsparseDestroyDnVec(vecX));  
    cusparseErrchk(hipsparseDestroyDnVec(vecY));  
}

void Preconditioner_block_ilu::apply_preconditioner(
    double *z_d,
    double *r_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle
){

    cudaErrchk(hipMemcpy(x_d, r_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToDevice))

    hipsparseDnVecDescr_t vecZ;
    cusparseErrchk(hipsparseCreateDnVec(
        &vecZ, rows_this_rank, z_d, HIP_R_64F
    ));

    const double one = 1.0;
    // Solve Ly = r
    cusparseErrchk(hipsparseSpSV_solve(default_cusparseHandle,
                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,
                        descr_L,
                        vecX,
                        vecY,
                        HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT,
                        spsvDescr_L,
                        buffer_L_d));

    // Solve L'z = y
    cusparseErrchk(hipsparseSpSV_solve(default_cusparseHandle,
                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        &one,
                        descr_Lt,
                        vecY,
                        vecZ,
                        HIP_R_64F,
                        HIPSPARSE_SPSV_ALG_DEFAULT,
                        spsvDescr_Lt,
                        buffer_Lt_d));

    cusparseErrchk(hipsparseDestroyDnVec(vecZ));  
}