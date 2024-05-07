#include "dist_objects.h"

Distributed_subblock::Distributed_subblock(
    int matrix_size,
    int *subblock_indices_local_h,
    int subblock_size,
    int *counts,
    int *displacements,
    int *counts_subblock,
    int *displacements_subblock,
    int nnz,
    double *data_d,
    int *row_ptr_compressed_d,
    int *col_indices_compressed_d,
    rocsparse_spmv_alg algo,
    MPI_Comm comm
){
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    this->matrix_size = matrix_size;
    this->subblock_size = subblock_size;
    this->nnz = nnz;
    this->comm = comm;

    this->counts = new int[size];
    this->displacements = new int[size];
    this->counts_subblock = new int[size];
    this->displacements_subblock = new int[size];

    for(int i = 0; i < size; i++){
        this->counts[i] = counts[i];
        this->displacements[i] = displacements[i];
        this->counts_subblock[i] = counts_subblock[i];
        this->displacements_subblock[i] = displacements_subblock[i];        
    }

    // subblock indices of the local vector part
    cudaErrchk(hipMalloc(&subblock_indices_local_d, counts_subblock[rank] * sizeof(int)));
    cudaErrchk(hipMemcpy(subblock_indices_local_d, subblock_indices_local_h, counts_subblock[rank] * sizeof(int), hipMemcpyHostToDevice));


    cudaErrchk(hipMalloc(&(this->data_d), nnz * sizeof(double)));
    cudaErrchk(hipMalloc(&(this->col_indices_compressed_d), nnz * sizeof(int)));
    cudaErrchk(hipMalloc(&(this->row_ptr_compressed_d), (counts_subblock[rank]+1) * sizeof(int)));
    cudaErrchk(hipMemcpy((this->data_d), data_d, nnz * sizeof(double), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemcpy((this->col_indices_compressed_d), col_indices_compressed_d, nnz * sizeof(int), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemcpy((this->row_ptr_compressed_d), row_ptr_compressed_d, (counts_subblock[rank]+1) * sizeof(int), hipMemcpyDeviceToDevice));

    // descriptor for subblock
    rocsparse_create_csr_descr(&descriptor,
                            counts_subblock[rank],
                            subblock_size,
                            nnz,
                            this->row_ptr_compressed_d,
                            this->col_indices_compressed_d,
                            this->data_d,
                            rocsparse_indextype_i32,
                            rocsparse_indextype_i32,
                            rocsparse_index_base_zero,
                            rocsparse_datatype_f64_r);

    double alpha = 1.0;
    double beta = 0.0;
    double *tmp_in_d;
    double *tmp_out_d;
    cudaErrchk(hipMalloc(&tmp_in_d, subblock_size * sizeof(double)));
    cudaErrchk(hipMemset(tmp_in_d, 1, subblock_size * sizeof(double))); //
    cudaErrchk(hipMalloc(&tmp_out_d, counts_subblock[rank] * sizeof(double)));

    rocsparse_dnvec_descr subblock_vector_descriptor_in;
    rocsparse_dnvec_descr subblock_vector_descriptor_out;

    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_in,
                                subblock_size,
                                tmp_in_d,
                                rocsparse_datatype_f64_r);

    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_out,
                                counts_subblock[rank],
                                tmp_out_d,
                                rocsparse_datatype_f64_r);

    rocsparse_handle rocsparse_handle;
    rocsparse_create_handle(&rocsparse_handle);

    this->algo = algo;

    rocsparse_spmv(rocsparse_handle,
                rocsparse_operation_none,
                &alpha,
                descriptor,
                subblock_vector_descriptor_in,
                &beta,
                subblock_vector_descriptor_out,
                rocsparse_datatype_f64_r,
                algo,
                &buffersize,
                nullptr);

    rocsparse_destroy_handle(rocsparse_handle);
    cudaErrchk(hipFree(tmp_in_d));
    cudaErrchk(hipFree(tmp_out_d));
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_in);
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_out);

    cudaErrchk(hipMalloc(&buffer_d, buffersize));

    events_recv = new hipEvent_t[size];
    streams_recv = new hipStream_t[size-1];
    send_requests = new MPI_Request[size-1];
    recv_requests = new MPI_Request[size-1];

    for(int i = 0; i < size; i++){
        cudaErrchk(hipEventCreateWithFlags(&events_recv[i], hipEventDisableTiming));
    }
    for(int i = 0; i < size-1; i++){
        cudaErrchk(hipStreamCreate(&streams_recv[i]));
    }

}

Distributed_subblock::~Distributed_subblock(){
    delete[] counts;
    delete[] displacements;
    delete[] counts_subblock;
    delete[] displacements_subblock;

    delete[] send_requests;
    delete[] recv_requests;

    for(int i = 0; i < size; i++){
        hipEventDestroy(events_recv[i]);
    }
    for(int i = 0; i < size-1; i++){
        hipStreamDestroy(streams_recv[i]);
    }

    delete[] events_recv;
    delete[] streams_recv;

    hipFree(subblock_indices_local_d);
    hipFree(data_d);
    hipFree(col_indices_compressed_d);
    hipFree(row_ptr_compressed_d);

    hipFree(buffer_d);
    rocsparse_destroy_spmat_descr(descriptor);
};