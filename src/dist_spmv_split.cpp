#include "dist_spmv.h"

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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        hipStreamSynchronize(default_stream);
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_d + A_subblock.displacements_subblock[rank], A_subblock.counts_subblock[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_d + A_subblock.displacements_subblock[source], A_subblock.counts_subblock[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_requests[i]);
        }
    }

    dspmv::manual_packing_overlap_compressed(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.recv_requests, MPI_STATUSES_IGNORE);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_compressed, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );     
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.send_requests, MPI_STATUSES_IGNORE);
    }

}


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
    rocsparse_handle &default_rocsparseHandle)
{
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;
    int flag;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(hipStreamSynchronize(default_stream));
        MPI_Iallgatherv(MPI_IN_PLACE, A_subblock.counts_subblock[rank],
            MPI_DOUBLE,
            p_subblock_d,
            A_subblock.counts_subblock,
            A_subblock.displacements_subblock,
            MPI_DOUBLE, A_distributed.comm, &A_subblock.send_requests[0]);
    }

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], default_stream);
        cudaErrchk(hipEventRecord(A_distributed.events_send[i], default_stream));
    }
    
    if(size > 1){
        MPI_Test(&A_subblock.send_requests[0], &flag, MPI_STATUS_IGNORE);
    }

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    if(size > 1){
        MPI_Test(&A_subblock.send_requests[0], &flag, MPI_STATUS_IGNORE);
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
            MPI_Wait(&A_distributed.recv_requests[i], MPI_STATUS_IGNORE);
            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors_compressed[i], A_distributed.recv_buffer_descriptor[i],
                &alpha, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algos_generic[i],
                &A_distributed.buffers_size_compressed[i],
                A_distributed.buffers_compressed_d[i]);
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
        
    }


    if(size > 1){
        MPI_Wait(&A_subblock.send_requests[0], MPI_STATUS_IGNORE);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_compressed, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );        

    if(size > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);
    }

}


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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        hipStreamSynchronize(default_stream);
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_d + A_subblock.displacements_subblock[rank], A_subblock.counts_subblock[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_d + A_subblock.displacements_subblock[source], A_subblock.counts_subblock[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_requests[i]);
        }
    }

    dspmv::manual_packing_overlap_compressed(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.recv_requests, MPI_STATUSES_IGNORE);
    }

    unpack(
        p_distributed.tot_vec_d, p_subblock_d,
        A_subblock.subblock_indices_d,
        A_subblock.subblock_size,
        default_stream
    );
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_uncompressed, p_distributed.descriptor,
        &alpha, vecAp_local, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    if(size > 1){
        MPI_Waitall(size-1, A_subblock.send_requests, MPI_STATUSES_IGNORE);
    }

}


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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        hipStreamSynchronize(default_stream);
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_d + A_subblock.displacements_subblock[rank], A_subblock.counts_subblock[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_d + A_subblock.displacements_subblock[source], A_subblock.counts_subblock[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_requests[i]);
        }
    }

    dspmv::manual_packing_singlekernel_compressed(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.recv_requests, MPI_STATUSES_IGNORE);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_compressed, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );     
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.send_requests, MPI_STATUSES_IGNORE);
    }

}


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
    rocsparse_handle &default_rocsparseHandle)
{
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;
    int flag;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(hipStreamSynchronize(default_stream));
        MPI_Iallgatherv(MPI_IN_PLACE, A_subblock.counts_subblock[rank],
            MPI_DOUBLE,
            p_subblock_d,
            A_subblock.counts_subblock,
            A_subblock.displacements_subblock,
            MPI_DOUBLE, A_distributed.comm, &A_subblock.send_requests[0]);
    }

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], default_stream);

        cudaErrchk(hipEventRecord(A_distributed.events_send[i], default_stream));
    }

    cudaErrchk(hipMemcpyAsync(
        A_distributed.p_compressed_d + A_distributed.displacements_compressed_h[0],
        p_distributed.vec_d[0],
        A_distributed.rows_this_rank * sizeof(double), hipMemcpyDeviceToDevice, default_stream));

    if(size > 1){
        MPI_Test(&A_subblock.send_requests[0], &flag, MPI_STATUS_IGNORE);
    }      

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    if(size > 1){
        MPI_Test(&A_subblock.send_requests[0], &flag, MPI_STATUS_IGNORE);
    }  

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = p_distributed.neighbours[i];
        int recv_tag = std::abs(recv_idx-A_distributed.rank);
        MPI_Irecv(A_distributed.p_compressed_d + A_distributed.displacements_compressed_h[i],
            A_distributed.nnz_cols_per_neighbour[i],
            MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i]);
    }
    
    if(size > 1){
        MPI_Test(&A_subblock.send_requests[0], &flag, MPI_STATUS_IGNORE);
    }   

    if(A_distributed.number_of_neighbours > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.recv_requests[1], MPI_STATUSES_IGNORE);    
    }   


    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_distributed.descriptor_compressed, A_distributed.p_compressed_descriptor,
        &beta, vecAp_local, rocsparse_datatype_f64_r,
        A_distributed.algo_generic,
        &A_distributed.buffer_size_compressed,
        A_distributed.buffer_compressed_d);

    if(size > 1){
        MPI_Wait(&A_subblock.send_requests[0], MPI_STATUS_IGNORE);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_compressed, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );        

    if(size > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);
    }

}


void manual_packing_singlekernel_compressed3(
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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);


    // post all send requests
    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        pack(A_subblock.send_buffer_d[i], p_subblock_d + A_subblock.displacements_subblock[rank],
            A_subblock.rows_per_neighbour_d[i], A_subblock.nnz_rows_per_neighbour[i], default_stream);

        cudaErrchk(hipEventRecord(A_subblock.events_send[i], default_stream));
    }


    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        int send_idx = A_subblock.neighbours[i];
        int send_tag = std::abs(send_idx-A_subblock.rank) + 2*size;

        cudaErrchk(hipEventSynchronize(A_subblock.events_send[i]));

        MPI_Isend(A_subblock.send_buffer_d[i], A_subblock.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_subblock.comm, &A_subblock.send_requests[i]);
    }

    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = A_subblock.neighbours[i];
        int recv_tag = std::abs(recv_idx-A_subblock.rank) + 2*size;
        MPI_Irecv(A_subblock.recv_buffer_d[i], A_subblock.nnz_cols_per_neighbour[i],
            MPI_DOUBLE, recv_idx, recv_tag, A_subblock.comm, &A_subblock.recv_requests[i]);
    }

    dspmv::manual_packing_singlekernel_compressed(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );

    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = A_subblock.neighbours[i];
        MPI_Wait(&A_subblock.recv_requests[i], MPI_STATUS_IGNORE);

        unpack(p_subblock_d + A_subblock.displacements_subblock[recv_idx],
            A_subblock.recv_buffer_d[i],
            A_subblock.cols_per_neighbour_d[i], A_subblock.nnz_cols_per_neighbour[i], default_stream);
    }

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_compressed, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );     
    if(size > 1){
        MPI_Waitall(A_subblock.number_of_neighbours-1,
            &A_subblock.send_requests[1], MPI_STATUSES_IGNORE);
    }

}

void manual_packing_singlekernel_compressed4(
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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        pack(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], default_stream);

        cudaErrchk(hipEventRecord(A_distributed.events_send[i], default_stream));
    }

    // pack dense sublblock p
    pack(A_subblock.p_double_compressed_d + A_subblock.displacements_compressed_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        pack(A_subblock.send_buffer_d[i], A_subblock.p_double_compressed_d + A_subblock.displacements_compressed_subblock[rank],
            A_subblock.rows_per_neighbour_d[i], A_subblock.nnz_rows_per_neighbour[i], default_stream);

        cudaErrchk(hipEventRecord(A_subblock.events_send[i], default_stream));
    }

    cudaErrchk(hipMemcpyAsync(
        A_distributed.p_compressed_d + A_distributed.displacements_compressed_h[0],
        p_distributed.vec_d[0],
        A_distributed.rows_this_rank * sizeof(double), hipMemcpyDeviceToDevice, default_stream));
    

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = p_distributed.neighbours[i];
        int recv_tag = std::abs(recv_idx-A_distributed.rank);
        MPI_Irecv(A_distributed.p_compressed_d + A_distributed.displacements_compressed_h[i],
            A_distributed.nnz_cols_per_neighbour[i],
            MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i]);
    }
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        // loop over neighbors
        int recv_idx = A_subblock.neighbours[i];
        int recv_tag = std::abs(recv_idx-A_subblock.rank) + 2*size;
        MPI_Irecv(A_subblock.p_double_compressed_d + A_subblock.displacements_compressed_subblock[recv_idx], 
            A_subblock.nnz_cols_per_neighbour[i],
            MPI_DOUBLE, recv_idx, recv_tag, A_subblock.comm, &A_subblock.recv_requests[i]);
    }
    for(int i = 1; i < A_subblock.number_of_neighbours; i++){
        int send_idx = A_subblock.neighbours[i];
        int send_tag = std::abs(send_idx-A_subblock.rank) + 2*size;

        cudaErrchk(hipEventSynchronize(A_subblock.events_send[i]));

        MPI_Isend(A_subblock.send_buffer_d[i], A_subblock.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_subblock.comm, &A_subblock.send_requests[i]);
    }

    if(A_distributed.number_of_neighbours > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.recv_requests[1], MPI_STATUSES_IGNORE);    
    }   
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_distributed.descriptor_compressed, A_distributed.p_compressed_descriptor,
        &beta, vecAp_local, rocsparse_datatype_f64_r,
        A_distributed.algo_generic,
        &A_distributed.buffer_size_compressed,
        A_distributed.buffer_compressed_d);

    if(size > 1){
        MPI_Waitall(A_subblock.number_of_neighbours-1,
            &A_subblock.recv_requests[1], MPI_STATUSES_IGNORE);
    }
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_double_compressed, A_subblock.p_double_compressed_descriptor,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_double_compressed,
        A_subblock.buffer_double_compressed_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream
    );     
    if(size > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1,
            &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);    
        MPI_Waitall(A_subblock.number_of_neighbours-1,
            &A_subblock.send_requests[1], MPI_STATUSES_IGNORE);
    }

}



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
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack(p_subblock_d + A_subblock.displacements_subblock[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.counts_subblock[rank],
        default_stream);

    if(size > 1){
        hipStreamSynchronize(default_stream);
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_d + A_subblock.displacements_subblock[rank], A_subblock.counts_subblock[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_d + A_subblock.displacements_subblock[source], A_subblock.counts_subblock[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_requests[i]);
        }
    }

    dspmv::manual_packing_singlekernel_compressed(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.recv_requests, MPI_STATUSES_IGNORE);
    }

    unpack(
        p_distributed.tot_vec_d, p_subblock_d,
        A_subblock.subblock_indices_d,
        A_subblock.subblock_size,
        default_stream
    );
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        A_subblock.descriptor_uncompressed, p_distributed.descriptor,
        &alpha, vecAp_local, rocsparse_datatype_f64_r,
        A_subblock.algo,
        &A_subblock.buffersize_compressed,
        A_subblock.buffer_compressed_d);
    if(size > 1){
        MPI_Waitall(size-1, A_subblock.send_requests, MPI_STATUSES_IGNORE);
    }

}


} // namespace dspmv_split_sparse