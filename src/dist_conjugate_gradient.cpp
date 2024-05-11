#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>

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
    Precon &precon)
{

    // initialize cuda

    // // set pointer mode to host
    // cublasErrchk(hipblasSetPointerMode(default_cublasHandle, HIPBLAS_POINTER_MODE_HOST)); // TEST @Manasa

    double a, b, na;
    double alpha, alpham1, r0;
    double    r_norm2_h[1];
    double    dot_h[1];
    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;
    double norm2_rhs = 0;


    //copy data to device
    // starting guess for p

    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemset(A_distributed.Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));


    //begin CG
    // norm of rhs for convergence check
    
    cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv(
        A_distributed,
        p_distributed,
        A_distributed.vecAp_local,
        A_distributed.default_stream,
        A_distributed.default_rocsparseHandle
    );

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &alpham1, A_distributed.Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    precon.apply_preconditioner(
        A_distributed.z_local_d,
        r_local_d,
        A_distributed.default_stream
    );
    
    cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, A_distributed.z_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(hipblasDscal(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &alpha, A_distributed.z_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(hipblasDcopy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, A_distributed.z_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        distributed_spmv(
            A_distributed,
            p_distributed,
            A_distributed.vecAp_local,
            A_distributed.default_stream,
            A_distributed.default_rocsparseHandle
        );

        cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, A_distributed.Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(hipblasDaxpy(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, &na, A_distributed.Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // Mz = r
        precon.apply_preconditioner(
            A_distributed.z_local_d,
            r_local_d,
            A_distributed.default_stream
        );

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(A_distributed.default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, A_distributed.z_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;

    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

}

template 
void preconditioned_conjugate_gradient<dspmv::alltoall_cam, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::alltoall_cam, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);

template 
void preconditioned_conjugate_gradient<dspmv::pointpoint_singlekernel_cam, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::pointpoint_singlekernel_cam, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);

template 
void preconditioned_conjugate_gradient<dspmv::pointpoint_singlekernel_cam2, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::pointpoint_singlekernel_cam2, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);

template 
void preconditioned_conjugate_gradient<dspmv::manual_packing, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::manual_packing, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);


template 
void preconditioned_conjugate_gradient<dspmv::manual_packing_cam, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::manual_packing_cam, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);


template 
void preconditioned_conjugate_gradient<dspmv::manual_packing_cam2, Preconditioner_none>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_none &precon);
template 
void preconditioned_conjugate_gradient<dspmv::manual_packing_cam2, Preconditioner_jacobi>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    Preconditioner_jacobi &precon);

} // namespace iterative_solver