#include "dist_objects.h"

Distributed_subblock::Distributed_subblock(
    int matrix_size,
    int *subblock_indices_local_h,
    int *subblock_indices_h,
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
    cudaErrchk(hipMalloc(&subblock_indices_d, subblock_size * sizeof(int)));
    cudaErrchk(hipMemcpy(subblock_indices_d, subblock_indices_h, subblock_size* sizeof(int), hipMemcpyHostToDevice));

    cudaErrchk(hipMalloc(&(this->data_d), nnz * sizeof(double)));
    cudaErrchk(hipMalloc(&(this->col_indices_compressed_d), nnz * sizeof(int)));
    cudaErrchk(hipMalloc(&(this->row_ptr_compressed_d), (counts_subblock[rank]+1) * sizeof(int)));
    cudaErrchk(hipMemcpy((this->data_d), data_d, nnz * sizeof(double), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemcpy((this->col_indices_compressed_d), col_indices_compressed_d, nnz * sizeof(int), hipMemcpyDeviceToDevice));
    cudaErrchk(hipMemcpy((this->row_ptr_compressed_d), row_ptr_compressed_d, (counts_subblock[rank]+1) * sizeof(int), hipMemcpyDeviceToDevice));

    cudaErrchk(hipMalloc(&(col_indices_uncompressed_d), nnz * sizeof(int)));
    cudaErrchk(hipMalloc(&(row_ptr_uncompressed_d), (matrix_size+1) * sizeof(int)));

    expand_row_ptr(
        row_ptr_uncompressed_d,
        row_ptr_compressed_d,
        subblock_indices_local_d,
        matrix_size,
        counts_subblock[rank]
    );
    expand_col_indices(
        col_indices_uncompressed_d,
        col_indices_compressed_d,
        subblock_indices_d,
        nnz,
        subblock_size
    );

    // descriptor_compressed for subblock
    rocsparse_create_csr_descr(&descriptor_compressed,
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
                descriptor_compressed,
                subblock_vector_descriptor_in,
                &beta,
                subblock_vector_descriptor_out,
                rocsparse_datatype_f64_r,
                algo,
                &buffersize_compressed,
                nullptr);

    cudaErrchk(hipFree(tmp_in_d));
    cudaErrchk(hipFree(tmp_out_d));
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_in);
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_out);

    cudaErrchk(hipMalloc(&buffer_compressed_d, buffersize_compressed));

    // descriptor_uncompressed for subblock
    rocsparse_create_csr_descr(&descriptor_uncompressed,
                            counts[rank],
                            matrix_size,
                            nnz,
                            this->row_ptr_uncompressed_d,
                            this->col_indices_uncompressed_d,
                            this->data_d,
                            rocsparse_indextype_i32,
                            rocsparse_indextype_i32,
                            rocsparse_index_base_zero,
                            rocsparse_datatype_f64_r);
    double *tmp_in_d_uncompressed;
    double *tmp_out_d_uncompressed;
    cudaErrchk(hipMalloc(&tmp_in_d_uncompressed, matrix_size * sizeof(double)));
    cudaErrchk(hipMalloc(&tmp_out_d_uncompressed, counts[rank] * sizeof(double)));

    rocsparse_dnvec_descr subblock_vector_descriptor_in_uncompressed;
    rocsparse_dnvec_descr subblock_vector_descriptor_out_uncompressed;

    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_in_uncompressed,
                                matrix_size,
                                tmp_in_d_uncompressed,
                                rocsparse_datatype_f64_r);
    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_out_uncompressed,
                                counts[rank],
                                tmp_out_d_uncompressed,
                                rocsparse_datatype_f64_r);
    rocsparse_spmv(rocsparse_handle,
                rocsparse_operation_none,
                &alpha,
                descriptor_uncompressed,
                subblock_vector_descriptor_in_uncompressed,
                &beta,
                subblock_vector_descriptor_out_uncompressed,
                rocsparse_datatype_f64_r,
                algo,
                &buffersize_uncompressed,
                nullptr);

    cudaErrchk(hipFree(tmp_in_d_uncompressed));
    cudaErrchk(hipFree(tmp_out_d_uncompressed));
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_in_uncompressed);
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_out_uncompressed);
    cudaErrchk(hipMalloc(&buffer_uncompressed_d, buffersize_uncompressed));


    rocsparse_destroy_handle(rocsparse_handle);

    events_send = new hipEvent_t[size];
    events_recv = new hipEvent_t[size];
    streams_recv = new hipStream_t[size-1];
    send_requests = new MPI_Request[size-1];
    recv_requests = new MPI_Request[size-1];

    for(int i = 0; i < size; i++){
        cudaErrchk(hipEventCreateWithFlags(&events_send[i], hipEventDisableTiming));
        cudaErrchk(hipEventCreateWithFlags(&events_recv[i], hipEventDisableTiming));
    }
    for(int i = 0; i < size-1; i++){
        cudaErrchk(hipStreamCreate(&streams_recv[i]));
    }

    analyze();

}

Distributed_subblock::~Distributed_subblock(){
    delete[] counts;
    delete[] displacements;
    delete[] counts_subblock;
    delete[] displacements_subblock;

    delete[] send_requests;
    delete[] recv_requests;

    for(int i = 0; i < size; i++){
        hipEventDestroy(events_send[i]);
        hipEventDestroy(events_recv[i]);
    }
    for(int i = 0; i < size-1; i++){
        hipStreamDestroy(streams_recv[i]);
    }

    delete[] events_send;
    delete[] events_recv;
    delete[] streams_recv;

    hipFree(subblock_indices_local_d);
    hipFree(data_d);
    hipFree(col_indices_compressed_d);
    hipFree(row_ptr_compressed_d);

    hipFree(buffer_compressed_d);
    rocsparse_destroy_spmat_descr(descriptor_compressed);

    delete[] neighbours;
    delete[] nnz_cols_per_neighbour;
    delete[] nnz_rows_per_neighbour;

    for(int i = 0; i < number_of_neighbours; i++){
        delete[] cols_per_neighbour_h[i];
        delete[] rows_per_neighbour_h[i];
        cudaErrchk(hipFree(cols_per_neighbour_d[i]));
        cudaErrchk(hipFree(rows_per_neighbour_d[i]));
    }

    delete[] cols_per_neighbour_h;
    delete[] cols_per_neighbour_d;
    delete[] rows_per_neighbour_h;
    delete[] rows_per_neighbour_d;

    for(int i = 0; i < number_of_neighbours; i++){
        cudaErrchk(hipFree(send_buffer_d[i]));
        cudaErrchk(hipFree(recv_buffer_d[i]));
    }
    delete[] send_buffer_d;
    delete[] recv_buffer_d;
};

void Distributed_subblock::analyze(){

    // goal is to find
    // nnz_cols_per_neighbour
    // nnz_rows_per_neighbour
    // cols_per_neighbour_d
    // rows_per_neighbour_d


    // first host approach to figure out best way
    int *row_ptr_compressed_h = new int[counts_subblock[rank]+1];
    int *col_indices_compressed_h = new int[nnz];
    cudaErrchk(hipMemcpy(row_ptr_compressed_h, row_ptr_compressed_d, (counts_subblock[rank]+1) * sizeof(int), hipMemcpyDeviceToHost));
    cudaErrchk(hipMemcpy(col_indices_compressed_h, col_indices_compressed_d, nnz * sizeof(int), hipMemcpyDeviceToHost));


    // mark rows and columns with non zero elements
    int *nz_cols = new int[subblock_size];
    int *nz_rows = new int[size*counts_subblock[rank]];
    for(int i = 0; i < subblock_size; i++){
        nz_cols[i] = 0;
    }
    for(int i = 0; i < size*counts_subblock[rank]; i++){
        nz_rows[i] = 0;
    }


    for(int i = 0; i < counts_subblock[rank]; i++){
        for(int j = row_ptr_compressed_h[i]; j < row_ptr_compressed_h[i+1]; j++){
            nz_cols[col_indices_compressed_h[j]] = 1;
        }
    }
    // arraz of 000..,111..., etc
    int *helper_array = new int[subblock_size];
    for(int i = 0; i < size; i++){
        for(int j = 0; j < counts_subblock[i]; j++){
            helper_array[displacements_subblock[i] + j] = i;
        }
    }
    for(int i = 0; i < counts_subblock[rank]; i++){
        for(int j = row_ptr_compressed_h[i]; j < row_ptr_compressed_h[i+1]; j++){
            nz_rows[i + counts_subblock[rank]*helper_array[col_indices_compressed_h[j]] ] = 1;
        }
    }
    // force your own piece to be one
    for(int i = 0; i < counts_subblock[rank]; i++){
        nz_cols[i + displacements_subblock[rank]] = 1;
        nz_rows[i + counts_subblock[rank]*rank] = 1;
    }
    int *nnz_cols_per_rank = new int[size];
    int *nnz_rows_per_rank = new int[size];

    for(int i = 0; i < size; i++){
        nnz_cols_per_rank[i] = 0;
        for(int j = 0; j < counts_subblock[i]; j++){
            if(nz_cols[displacements_subblock[i] +  j] > 0){
                nnz_cols_per_rank[i]++;
            }
        }
    }
    

    for(int i = 0; i < size; i++){
        nnz_rows_per_rank[i] = 0;
        for(int j = 0; j < counts_subblock[rank]; j++){
            if(nz_rows[i*counts_subblock[rank] +  j] > 0){
                nnz_rows_per_rank[i]++;
            }
        }
    }
    number_of_neighbours = 0;
    int number_of_neighbours2 = 0;
    
    for(int i = 0; i < size; i++){
        if(nnz_cols_per_rank[i] > 0){
            number_of_neighbours++;
        }
        if(nnz_rows_per_rank[i] > 0){
            number_of_neighbours2++;
        }
    }
    if(number_of_neighbours2 != number_of_neighbours){
        printf("Error in number of neighbours\n");
        exit(1);
    }
    neighbours = new int[number_of_neighbours];
    int count_neighbours = 0;
    for(int i = 0; i < size; i++){
        int idx = (rank + i) % size;
        if(nnz_cols_per_rank[idx] > 0){
            neighbours[count_neighbours] = idx;
            count_neighbours++;
        }
    }
    nnz_cols_per_neighbour = new int[number_of_neighbours];
    nnz_rows_per_neighbour = new int[number_of_neighbours];
    for(int i = 0; i < number_of_neighbours; i++){
        nnz_cols_per_neighbour[i] = nnz_cols_per_rank[neighbours[i]];
        nnz_rows_per_neighbour[i] = nnz_rows_per_rank[neighbours[i]];
        // std::cout << rank << " " << i << " " << neighbours[i] << " " << nnz_cols_per_neighbour[i] << " " << nnz_rows_per_neighbour[i] << std::endl;
    }
    // std::cout << rank << " " << counts_subblock[rank] << std::endl;
    // MPI_Barrier(comm);
    // exit(1);

    cols_per_neighbour_h = new int*[number_of_neighbours];
    rows_per_neighbour_h = new int*[number_of_neighbours];
    for(int i = 0; i < number_of_neighbours; i++){
        cols_per_neighbour_h[i] = new int[nnz_cols_per_neighbour[i]];
        rows_per_neighbour_h[i] = new int[nnz_rows_per_neighbour[i]];
    }
    for(int i = 0; i < number_of_neighbours; i++){
        int idx = neighbours[i];
        int count_cols = 0;
        int count_rows = 0;
        // todo do on the gpu with copy if
        for(int j = 0; j < counts_subblock[idx]; j++){
            if(nz_cols[displacements_subblock[idx] +  j] > 0){
                cols_per_neighbour_h[i][count_cols] = j;
                count_cols++;
            }
        }
        for(int j = 0; j < counts_subblock[rank]; j++){
            if(nz_rows[idx*counts_subblock[rank] +  j] > 0){
                rows_per_neighbour_h[i][count_rows] = j;
                count_rows++;
            }
        }
    }
    cols_per_neighbour_d = new int*[number_of_neighbours];
    rows_per_neighbour_d = new int*[number_of_neighbours];
    for(int i = 0; i < number_of_neighbours; i++){
        cudaErrchk(hipMalloc(&cols_per_neighbour_d[i], nnz_cols_per_neighbour[i] * sizeof(int)));
        cudaErrchk(hipMalloc(&rows_per_neighbour_d[i], nnz_rows_per_neighbour[i] * sizeof(int)));
        cudaErrchk(hipMemcpy(cols_per_neighbour_d[i], cols_per_neighbour_h[i], nnz_cols_per_neighbour[i] * sizeof(int), hipMemcpyHostToDevice));
        cudaErrchk(hipMemcpy(rows_per_neighbour_d[i], rows_per_neighbour_h[i], nnz_rows_per_neighbour[i] * sizeof(int), hipMemcpyHostToDevice));
    }

    send_buffer_d = new double*[number_of_neighbours];
    recv_buffer_d = new double*[number_of_neighbours];
    for(int i = 0; i < number_of_neighbours; i++){
        cudaErrchk(hipMalloc(&send_buffer_d[i], nnz_rows_per_neighbour[i] * sizeof(double)));
        cudaErrchk(hipMalloc(&recv_buffer_d[i], nnz_cols_per_neighbour[i] * sizeof(double)));
    }




    delete[] row_ptr_compressed_h;
    delete[] col_indices_compressed_h;

    delete[] nz_cols;
    delete[] nz_rows;

    delete[] helper_array;
    delete[] nnz_cols_per_rank;
    delete[] nnz_rows_per_rank;

}