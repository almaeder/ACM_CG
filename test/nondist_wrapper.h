#pragma once
#include "../src/dist_conjugate_gradient.h"
#include "../src/dist_spmv.h"
#include "../src/preconditioner.h"
#include <hip/hip_runtime.h>
#include "utils_gpu.h"
#include <hipblas.h>
#include "utils.h"

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, rocsparse_dnvec_descr&, hipStream_t&, rocsparse_handle&), typename Precon>
void test_preconditioned(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *starting_guess_h,
    double *test_solution_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken,
    int number_of_measurements);

template <void (*distributed_spmv_split_sparse)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    rocsparse_dnvec_descr &,
    Distributed_vector &,
    double *,
    rocsparse_dnvec_descr &,
    rocsparse_dnvec_descr &,
    double *,
    hipStream_t &,
    rocsparse_handle &)>
void test_preconditioned_split_sparse(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *subblock_data_h,
    int *subblock_col_indices_h,
    int *subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *starting_guess_h,
    double *test_solution_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken,
    int number_of_measurements);