#include "nondist_wrapper.h"

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, rocsparse_dnvec_descr&, hipStream_t&, rocsparse_handle&)>
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
    int number_of_measurements)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0){
        std::cout << "Start test" << std::endl;
    }
    MPI_Barrier(comm);
    int counts[size];
    int displacements[size];
    int rows_this_rank;    
    split_matrix(matrix_size, size, counts, displacements);
    int row_start_index = displacements[rank];
    rows_this_rank = counts[rank];

    // Split full matrix into rows
    int *row_indptr_local_h = new int[rows_this_rank+1];
    double *r_local_h = new double[rows_this_rank];
    for (int i = 0; i < rows_this_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_this_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_this_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];
    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    rocsparse_spmv_alg algos[size];
    for(int k = 0; k < size; k++){
        algos[k] = rocsparse_spmv_alg_csr_stream;
    }        
    algos[rank] = rocsparse_spmv_alg_csr_adaptive;
    // use class contructor where whole rows_this_rank*matrix_size is input
    // possible to directly give number_of_neighbors * (rows_this_rank*rows_other_rank) as input
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        algos,
        comm
    );
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );

    double *r_local_d;
    double *x_local_d;
    hipMalloc(&r_local_d, rows_this_rank * sizeof(double));
    hipMalloc(&x_local_d, rows_this_rank * sizeof(double));
    hipMemcpy(r_local_d, r_local_h, rows_this_rank * sizeof(double), hipMemcpyHostToDevice);

    

    for(int i = 0; i < number_of_measurements; i++){
        // reset starting guess and right hand side
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_this_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(r_local_d, r_local_h, rows_this_rank * sizeof(double), hipMemcpyHostToDevice);

        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_start = std::chrono::high_resolution_clock::now();

        Preconditioner_jacobi precon(A_distributed);
        iterative_solver::preconditioned_conjugate_gradient<dspmv::gpu_packing, Preconditioner_jacobi>(
            A_distributed,
            p_distributed,
            r_local_d,
            x_local_d,
            relative_tolerance,
            max_iterations,
            comm,
            precon);


        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_end = std::chrono::high_resolution_clock::now();
        time_taken[i] += std::chrono::duration<double>(time_end - time_start).count();
        if(rank == 0){
            std::cout << rank << " time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }
    //copy solution to host
    cudaErrchk(hipMemcpy(test_solution_h + row_start_index,
        x_local_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToHost));
    // MPI allgatherv of the solution inplace
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, test_solution_h, counts, displacements, MPI_DOUBLE, comm);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);
}
template 
void test_preconditioned<dspmv::gpu_packing>(
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
template 
void test_preconditioned<dspmv::gpu_packing_cam>(
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
    int number_of_measurements)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0){
        std::cout << "Start test split" << std::endl;
    }
    MPI_Barrier(comm);
    int counts[size];
    int displacements[size];
    int rows_this_rank;    
    split_matrix(matrix_size, size, counts, displacements);
    int row_start_index = displacements[rank];
    rows_this_rank = counts[rank];

    // Split sparse matrix part into rows
    int *row_indptr_local_h = new int[rows_this_rank+1];
    double *r_local_h = new double[rows_this_rank];
    #pragma omp parallel for
    for (int i = 0; i < rows_this_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    #pragma omp parallel
    for (int i = 0; i < rows_this_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_this_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];
    #pragma omp parallel for
    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    rocsparse_spmv_alg algos[size];
    for(int k = 0; k < size; k++){
        algos[k] = rocsparse_spmv_alg_csr_stream;
    }        
    algos[rank] = rocsparse_spmv_alg_csr_adaptive;

    // use class contructor where whole rows_this_rank*matrix_size is input
    // possible to directly give number_of_neighbors * (rows_this_rank*rows_other_rank) as input
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        algos,
        comm
    );

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );


    double *r_local_d;
    double *x_local_d;
    hipMalloc(&r_local_d, rows_this_rank * sizeof(double));
    hipMalloc(&x_local_d, rows_this_rank * sizeof(double));

    int *subblock_indices_d;
    hipMalloc(&subblock_indices_d, subblock_size * sizeof(int));
    hipMemcpy(subblock_indices_d, subblock_indices_h, subblock_size * sizeof(int), hipMemcpyHostToDevice);

    int *count_subblock = new int[size];
    int *displ_subblock = new int[size];

    // split subblock between ranks
    for(int i = 0; i < size; i++){
        count_subblock[i] = 0;
    }
    #pragma omp parallel for
    for(int j = 0; j < size; j++){
        int tmp = 0;
        for(int i = 0; i < subblock_size; i++){
            if( subblock_indices_h[i] >= displacements[j] && subblock_indices_h[i] < displacements[j] + counts[j]){
                tmp++;
            }
        }
        count_subblock[j] = tmp;
    }
    displ_subblock[0] = 0;
    for(int i = 1; i < size; i++){
        displ_subblock[i] = displ_subblock[i-1] + count_subblock[i-1];
    }

    // subblock indices of the local vector part
    int *subblock_indices_local_h = new int[count_subblock[rank]];
    #pragma omp parallel for
    for(int i = 0; i < count_subblock[rank]; i++){
        subblock_indices_local_h[i] = subblock_indices_h[displ_subblock[rank] + i] - displacements[rank];
    }

    int *subblock_indices_local_d;
    hipMalloc(&subblock_indices_local_d, count_subblock[rank] * sizeof(int));
    hipMemcpy(subblock_indices_local_d, subblock_indices_local_h, count_subblock[rank] * sizeof(int), hipMemcpyHostToDevice);


    // split subblock by rows
    int *A_subblock_row_ptr_local_h = new int[count_subblock[rank]+1];
    #pragma omp parallel for
    for (int i = 0; i < count_subblock[rank]+1; ++i) {
        A_subblock_row_ptr_local_h[i] = subblock_row_ptr_h[i+displ_subblock[rank]] - subblock_row_ptr_h[displ_subblock[rank]];
    }

    int nnz_local_subblock = A_subblock_row_ptr_local_h[count_subblock[rank]];
    double *A_subblock_data_local_h = new double[nnz_local_subblock];
    int *A_subblock_col_indices_local_h = new int[nnz_local_subblock];
    #pragma omp parallel for
    for (int i = 0; i < nnz_local_subblock; ++i) {
        A_subblock_col_indices_local_h[i] = subblock_col_indices_h[i+subblock_row_ptr_h[displ_subblock[rank]]];
        A_subblock_data_local_h[i] = subblock_data_h[i+subblock_row_ptr_h[displ_subblock[rank]]];
    }

    double *A_subblock_data_local_d;
    int *A_subblock_col_indices_local_d;
    int *A_subblock_row_ptr_local_d;
    hipMalloc(&A_subblock_data_local_d, nnz_local_subblock * sizeof(double));
    hipMalloc(&A_subblock_col_indices_local_d, nnz_local_subblock * sizeof(int));
    hipMalloc(&A_subblock_row_ptr_local_d, (count_subblock[rank]+1) * sizeof(int));
    hipMemcpy(A_subblock_data_local_d, A_subblock_data_local_h, nnz_local_subblock * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_col_indices_local_d, A_subblock_col_indices_local_h, nnz_local_subblock * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_row_ptr_local_d, A_subblock_row_ptr_local_h, (count_subblock[rank]+1) * sizeof(int), hipMemcpyHostToDevice);

    // TODO class for subblock
    // descriptor for subblock
    rocsparse_spmat_descr subblock_descriptor;
    rocsparse_create_csr_descr(&subblock_descriptor,
                            count_subblock[rank],
                            subblock_size,
                            nnz_local_subblock,
                            A_subblock_row_ptr_local_d,
                            A_subblock_col_indices_local_d,
                            A_subblock_data_local_d,
                            rocsparse_indextype_i32,
                            rocsparse_indextype_i32,
                            rocsparse_index_base_zero,
                            rocsparse_datatype_f64_r);

    double alpha = 1.0;
    double beta = 0.0;
    double *tmp_in_d;
    double *tmp_out_d;
    hipMalloc(&tmp_in_d, subblock_size * sizeof(double));
    hipMemset(tmp_in_d, 1, subblock_size * sizeof(double)); //
    hipMalloc(&tmp_out_d, count_subblock[rank] * sizeof(double));

    rocsparse_dnvec_descr subblock_vector_descriptor_in;
    rocsparse_dnvec_descr subblock_vector_descriptor_out;

    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_in,
                                subblock_size,
                                tmp_in_d,
                                rocsparse_datatype_f64_r);

    // Create dense vector Y
    rocsparse_create_dnvec_descr(&subblock_vector_descriptor_out,
                                count_subblock[rank],
                                tmp_out_d,
                                rocsparse_datatype_f64_r);

    size_t subblock_buffersize;

    rocsparse_handle rocsparse_handle;
    rocsparse_create_handle(&rocsparse_handle);


    rocsparse_spmv_alg algo = rocsparse_spmv_alg_csr_adaptive;

    rocsparse_spmv(rocsparse_handle,
                rocsparse_operation_none,
                &alpha,
                subblock_descriptor,
                subblock_vector_descriptor_in,
                &beta,
                subblock_vector_descriptor_out,
                rocsparse_datatype_f64_r,
                algo,
                &subblock_buffersize,
                nullptr);
    
    rocsparse_destroy_handle(rocsparse_handle);

    hipFree(tmp_in_d);
    hipFree(tmp_out_d);

    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_in);
    rocsparse_destroy_dnvec_descr(subblock_vector_descriptor_out);

    double *subblock_buffer_d;
    hipMalloc(&subblock_buffer_d, subblock_buffersize);


    Distributed_subblock_sparse A_subblock;
    A_subblock.subblock_indices_local_d = subblock_indices_local_d;
    A_subblock.descriptor = &subblock_descriptor;
    A_subblock.algo = algo;
    A_subblock.buffersize = &subblock_buffersize;
    A_subblock.buffer_d = subblock_buffer_d;
    A_subblock.subblock_size = subblock_size;
    A_subblock.count_subblock_h = count_subblock;
    A_subblock.displ_subblock_h = displ_subblock;
    A_subblock.send_subblock_requests = new MPI_Request[size-1];
    A_subblock.recv_subblock_requests = new MPI_Request[size-1];
    A_subblock.streams_recv_subblock = new hipStream_t[size-1];



    for(int i = 0; i < size-1; i++){
        hipStreamCreate(&A_subblock.streams_recv_subblock[i]);
    }
    A_subblock.events_recv_subblock = new hipEvent_t[size];
    for(int i = 0; i < size; i++){
        hipEventCreateWithFlags(&A_subblock.events_recv_subblock[i], hipEventDisableTiming);
    }

    for(int i = 0; i < number_of_measurements; i++){

        hipMemcpy(r_local_d, r_local_h, rows_this_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_this_rank * sizeof(double), hipMemcpyHostToDevice);


        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_start = std::chrono::high_resolution_clock::now();

        Preconditioner_jacobi_split precon(
            A_distributed,
            count_subblock[rank],
            subblock_indices_local_d,
            A_subblock_row_ptr_local_d,
            A_subblock_col_indices_local_d,
            A_subblock_data_local_d,
            displ_subblock[rank]);
        
        iterative_solver::preconditioned_conjugate_gradient_split<distributed_spmv_split_sparse, Preconditioner_jacobi_split>(
            A_subblock,
            A_distributed,
            p_distributed,
            r_local_d,
            x_local_d,
            relative_tolerance,
            max_iterations,
            comm,
            precon);

        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_end = std::chrono::high_resolution_clock::now();
        time_taken[i] += std::chrono::duration<double>(time_end - time_start).count();
        if(rank == 0){
            std::cout << rank << " time_taken["<<i<<"] " << time_taken[i] << std::endl;
        }
    }


    for(int i = 0; i < size-1; i++){
        hipStreamDestroy(A_subblock.streams_recv_subblock[i]);
    }


    //copy solution to host
    cudaErrchk(hipMemcpy(test_solution_h + row_start_index,
        x_local_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToHost));
    // MPI allgatherv of the solution inplace
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, test_solution_h, counts, displacements, MPI_DOUBLE, comm);

    for(int i = 0; i < size; i++){
        hipEventDestroy(A_subblock.events_recv_subblock[i]);
    }

    delete[] A_subblock.streams_recv_subblock;


    delete[] A_subblock.send_subblock_requests;
    delete[] A_subblock.recv_subblock_requests;    
    delete[] A_subblock.events_recv_subblock;

    delete[] count_subblock;
    delete[] displ_subblock;
    delete[] subblock_indices_local_h;
    hipFree(subblock_indices_local_d);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);
    hipFree(subblock_indices_d);

    delete[] A_subblock_row_ptr_local_h;
    delete[] A_subblock_data_local_h;
    delete[] A_subblock_col_indices_local_h;
    hipFree(A_subblock_data_local_d);
    hipFree(A_subblock_col_indices_local_d);
    hipFree(A_subblock_row_ptr_local_d);

    hipFree(subblock_buffer_d);

}

template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *subblock_data_h,
    int *subblock_col_indices_h,
    int *subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int number_of_measurements);
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse2>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *subblock_data_h,
    int *subblock_col_indices_h,
    int *subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int number_of_measurements);
template 
void test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse3>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *subblock_data_h,
    int *subblock_col_indices_h,
    int *subblock_row_ptr_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken, int number_of_measurements);
