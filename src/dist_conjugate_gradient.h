#pragma once
#include <string> 
#include <omp.h>
#include "utils_cg.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <hipblas.h>
#include <iostream>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include <unistd.h>  
#include "rocsparse.h"
#include "preconditioner.h"

namespace iterative_solver{

template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle&),
    typename Precon>
void preconditioned_conjugate_gradient(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Precon &precon);

template <void (*distributed_spmv_split_sparse)
    (Distributed_subblock_sparse &,
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
    rocsparse_handle &),
    typename Precon>
void preconditioned_conjugate_gradient_split(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Precon &precon);

} // namespace iterative_solver