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

    int *counts_subblock = new int[size];
    int *displacements_subblock = new int[size];

    // split subblock between ranks
    for(int i = 0; i < size; i++){
        counts_subblock[i] = 0;
    }
    #pragma omp parallel for
    for(int j = 0; j < size; j++){
        int tmp = 0;
        for(int i = 0; i < subblock_size; i++){
            if( subblock_indices_h[i] >= displacements[j] && subblock_indices_h[i] < displacements[j] + counts[j]){
                tmp++;
            }
        }
        counts_subblock[j] = tmp;
    }
    displacements_subblock[0] = 0;
    for(int i = 1; i < size; i++){
        displacements_subblock[i] = displacements_subblock[i-1] + counts_subblock[i-1];
    }

    // subblock indices of the local vector part
    int *subblock_indices_local_h = new int[counts_subblock[rank]];
    #pragma omp parallel for
    for(int i = 0; i < counts_subblock[rank]; i++){
        subblock_indices_local_h[i] = subblock_indices_h[displacements_subblock[rank] + i] - displacements[rank];
    }

    // split subblock by rows
    int *A_subblock_row_ptr_local_h = new int[counts_subblock[rank]+1];
    #pragma omp parallel for
    for (int i = 0; i < counts_subblock[rank]+1; ++i) {
        A_subblock_row_ptr_local_h[i] = subblock_row_ptr_h[i+displacements_subblock[rank]] - subblock_row_ptr_h[displacements_subblock[rank]];
    }

    int nnz_local_subblock = A_subblock_row_ptr_local_h[counts_subblock[rank]];
    double *A_subblock_data_local_h = new double[nnz_local_subblock];
    int *A_subblock_col_indices_local_h = new int[nnz_local_subblock];
    #pragma omp parallel for
    for (int i = 0; i < nnz_local_subblock; ++i) {
        A_subblock_col_indices_local_h[i] = subblock_col_indices_h[i+subblock_row_ptr_h[displacements_subblock[rank]]];
        A_subblock_data_local_h[i] = subblock_data_h[i+subblock_row_ptr_h[displacements_subblock[rank]]];
    }

    double *A_subblock_data_local_d;
    int *A_subblock_col_indices_local_d;
    int *A_subblock_row_ptr_local_d;
    hipMalloc(&A_subblock_data_local_d, nnz_local_subblock * sizeof(double));
    hipMalloc(&A_subblock_col_indices_local_d, nnz_local_subblock * sizeof(int));
    hipMalloc(&A_subblock_row_ptr_local_d, (counts_subblock[rank]+1) * sizeof(int));
    hipMemcpy(A_subblock_data_local_d, A_subblock_data_local_h, nnz_local_subblock * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_col_indices_local_d, A_subblock_col_indices_local_h, nnz_local_subblock * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(A_subblock_row_ptr_local_d, A_subblock_row_ptr_local_h, (counts_subblock[rank]+1) * sizeof(int), hipMemcpyHostToDevice);


    Distributed_subblock A_subblock(
        matrix_size,
        subblock_indices_local_h,
        subblock_indices_h,
        subblock_size,
        counts,
        displacements,
        counts_subblock,
        displacements_subblock,
        nnz_local_subblock,
        A_subblock_data_local_d,
        A_subblock_row_ptr_local_d,
        A_subblock_col_indices_local_d,
        rocsparse_spmv_alg_csr_adaptive,
        comm
    );

    for(int i = 0; i < number_of_measurements; i++){

        hipMemcpy(r_local_d, r_local_h, rows_this_rank * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(x_local_d, starting_guess_h + row_start_index,
            rows_this_rank * sizeof(double), hipMemcpyHostToDevice);


        hipDeviceSynchronize();
        MPI_Barrier(MPI_COMM_WORLD);
        auto time_start = std::chrono::high_resolution_clock::now();

        Preconditioner_jacobi_split precon(
            A_distributed,
            A_subblock);
        
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


    //copy solution to host
    cudaErrchk(hipMemcpy(test_solution_h + row_start_index,
        x_local_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToHost));
    // MPI allgatherv of the solution inplace
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, test_solution_h, counts, displacements, MPI_DOUBLE, comm);


    delete[] counts_subblock;
    delete[] displacements_subblock;
    delete[] subblock_indices_local_h;

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);

    delete[] A_subblock_row_ptr_local_h;
    delete[] A_subblock_data_local_h;
    delete[] A_subblock_col_indices_local_h;
    hipFree(A_subblock_data_local_d);
    hipFree(A_subblock_col_indices_local_d);
    hipFree(A_subblock_row_ptr_local_d);
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
