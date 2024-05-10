#include "dist_spmv.h"


namespace dspmv{

void alltoall_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
){
    double alpha = 1.0;
    double beta = 0.0;

    cudaErrchk(hipStreamSynchronize(default_stream));
    MPI_Allgatherv(p_distributed.vec_d[0],
        A_distributed.rows_this_rank, MPI_DOUBLE,
        p_distributed.tot_vec_d, A_distributed.counts, A_distributed.displacements, MPI_DOUBLE, A_distributed.comm);   
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_distributed.descriptor, p_distributed.descriptor,
        &beta, vecAp_local, rocsparse_datatype_f64_r,
        A_distributed.algo_generic,
        &A_distributed.buffer_size,
        A_distributed.buffer_d);                 

}

void pointpoint_singlekernel_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
){
    double alpha = 1.0;
    double beta = 0.0;

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], default_stream);

        cudaErrchk(hipEventRecord(A_distributed.events_send[i], default_stream));
    }

    cudaErrchk(hipMemcpyAsync(p_distributed.tot_vec_d + A_distributed.displacements[A_distributed.rank],
        p_distributed.vec_d[0],
        A_distributed.rows_this_rank * sizeof(double), hipMemcpyDeviceToDevice, default_stream));
    

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = p_distributed.neighbours[i];
        int recv_tag = std::abs(recv_idx-A_distributed.rank);
        MPI_Irecv(A_distributed.recv_buffer_d[i], A_distributed.nnz_cols_per_neighbour[i],
            MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i]);
    }
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = p_distributed.neighbours[i];
        MPI_Wait(&A_distributed.recv_requests[i], MPI_STATUS_IGNORE);

        unpack(p_distributed.tot_vec_d + A_distributed.displacements[recv_idx],
            A_distributed.recv_buffer_d[i],
            A_distributed.cols_per_neighbour_d[i], A_distributed.nnz_cols_per_neighbour[i], default_stream);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_distributed.descriptor, p_distributed.descriptor,
        &beta, vecAp_local, rocsparse_datatype_f64_r,
        A_distributed.algo_generic,
        &A_distributed.buffer_size,
        A_distributed.buffer_d);

    if(A_distributed.number_of_neighbours > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);    
    }        
}


void manual_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle
){

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    cudaErrchk(hipEventRecord(A_distributed.event_default_finished, default_stream));

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(hipStreamWaitEvent(A_distributed.streams_send[i], A_distributed.event_default_finished, 0));
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(hipMemcpyAsync(A_distributed.send_buffer_h[i], A_distributed.send_buffer_d[i],
            A_distributed.nnz_rows_per_neighbour[i] * sizeof(double), hipMemcpyDeviceToHost, A_distributed.streams_send[i]));
    
        cudaErrchk(hipEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }
    
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_h[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx-A_distributed.rank);
            MPI_Irecv(A_distributed.recv_buffer_h[i+1], A_distributed.nnz_cols_per_neighbour[i+1],
                MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        // calc A*p
        if(i > 0){
            cudaErrchk(hipStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }
        if(i > 0){
            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algos_generic[i],
                &A_distributed.buffers_size[i],
                A_distributed.buffers_d[i]);

        }
        else{
            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algos_generic[i],
                &A_distributed.buffers_size[i],
                A_distributed.buffers_d[i]);
        }

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&A_distributed.recv_requests[i+1], MPI_STATUS_IGNORE);

            cudaErrchk(hipMemcpyAsync(A_distributed.recv_buffer_d[i+1], A_distributed.recv_buffer_h[i+1],
                A_distributed.nnz_cols_per_neighbour[i+1] * sizeof(double), hipMemcpyHostToDevice, A_distributed.streams_recv[i+1]));

            unpack(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(hipEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

}


void manual_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    // cuda aware mpi
    cudaErrchk(hipEventRecord(A_distributed.event_default_finished, default_stream));

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(hipStreamWaitEvent(A_distributed.streams_send[i], A_distributed.event_default_finished, 0));
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(hipEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }
    

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx-A_distributed.rank);
            MPI_Irecv(A_distributed.recv_buffer_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1],
                MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        // calc A*p
        if(i > 0){
            cudaErrchk(hipStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }
        if(i > 0){
            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algos_generic[i],
                &A_distributed.buffers_size[i],
                A_distributed.buffers_d[i]);
        }
        else{
            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algos_generic[i],
                &A_distributed.buffers_size[i],
                A_distributed.buffers_d[i]);
        }

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&A_distributed.recv_requests[i+1], MPI_STATUS_IGNORE);

            unpack(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(hipEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

}

} // namespace dspmv