#pragma once
#include <hip/hip_runtime.h>

void extract_diagonal_inv(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv,
    int matrix_size
);

void extract_diagonal(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values,
    int matrix_size
);

void  extract_add_subblock_diagonal(
    int *subblock_indices_d,
    int *subblock_row_ptr_d,
    int *subblock_col_indices_d,
    double *subblock_data_d,
    double *diag_inv_d,
    int subblock_rows_size,
    int displ_subblock_this_rank
);

void inv_inplace(
    double *data_d,
    int size
);

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements);

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream);

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements);

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream);

void unpack_add(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements);

void unpack_add(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream);

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n);

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n,
    hipStream_t stream);

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n);

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    hipStream_t stream);

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n);

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    hipStream_t stream);

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size);

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size,
    hipStream_t stream);