
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

void alltoall(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void pointpoint_overlap(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_overlap(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_overlap_compressed(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);


void pointpoint_singlekernel(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_singlekernel(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);

void manual_packing_singlekernel_compressed(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle);


} // namespace dspmv

namespace dspmv_split_sparse{

void manual_packing_overlap_compressed1(
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

void manual_packing_overlap_compressed2(
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

void uncompressed_manual_packing(
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

void manual_packing_singlekernel_compressed1(
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

void manual_packing_singlekernel_compressed2(
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

void uncompressed_manual_singlekernel(
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