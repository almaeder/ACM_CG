
#pragma once
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include "utils_cg.h"
#include <pthread.h>
#include "rocsparse.h"

namespace dspmv{

void alltoall_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void pointpoint_singlekernel_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_cam2(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

} // namespace dspmv

namespace dspmv_split_sparse{

void spmm_split_sparse1(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    rocsparse_dnvec_descr &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void spmm_split_sparse2(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    rocsparse_dnvec_descr &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void spmm_split_sparse3(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    rocsparse_dnvec_descr &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void spmm_split_sparse4(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    rocsparse_dnvec_descr &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

} // namespace dspmv_split_sparse