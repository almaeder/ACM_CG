#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "utils_gpu.h"
#include <hipblas.h>
#include "../src/dist_conjugate_gradient.h"
#include "../src/dist_spmv.h"
#include "../src/preconditioner.h"

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
        hipMemcpy(x_local_d, starting_guess_h,
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
    cudaErrchk(hipMemcpy(test_solution_h, x_local_d, rows_this_rank * sizeof(double), hipMemcpyDeviceToHost));


    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    hipFree(r_local_d);
    hipFree(x_local_d);

    MPI_Barrier(comm);
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


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hipError_t set_device_error = hipSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    int matsize = 100;
    std::string data_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices";
    std::string save_path ="/scratch/project_465000929/maederal/ACM_Poster/results/";

    int matrix_size;
    int nnz;     
    load_binary_array<int>(data_path + "/K_nnz"+ "_"+ std::to_string(matsize)+".bin", &nnz, 1);
    load_binary_array<int>(data_path + "/K_matrix_size"+ "_"+ std::to_string(matsize)+".bin", &matrix_size, 1);

    if(rank == 0){
        std::cout << "matrix_size " << matrix_size << std::endl;
        std::cout << "nnz " << nnz << std::endl;
    }

    int start_up_measurements = 0;
    int true_number_of_measurements = 10;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;

    int max_iterations = 10000;
    double relative_tolerance = 1.46524e-09;

    int counts[size];
    int displacements[size];
    int rows_this_rank;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_this_rank = counts[rank];
    int row_end_index = row_start_index + rows_this_rank;

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " row_end_index " << row_end_index << std::endl;
    std::cout << "rank " << rank << " rows_this_rank " << rows_this_rank << std::endl;

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];


    std::string data_filename;
    std::string row_ptr_filename;
    std::string col_indices_filename;
    std::string rhs_filename;
    std::string solution_filename;

    std::cout << "rank " << rank << " data_path " << data_path << std::endl;

    data_filename = data_path + "/K_data_"+std::to_string(matsize)+".bin";
    row_ptr_filename = data_path + "/K_row_ptr_"+std::to_string(matsize)+".bin";
    col_indices_filename = data_path + "/K_col_indices_"+std::to_string(matsize)+".bin";
    rhs_filename = data_path + "/K_rhs_"+std::to_string(matsize)+".bin";
    solution_filename = data_path + "/K_solution_"+std::to_string(matsize)+".bin";


    std::cout << "rank " << rank << " Loading data" << std::endl;
    std::cout << "rank " << rank << " data_filename " << data_filename << std::endl;
    std::cout << "rank " << rank << " row_ptr_filename " << row_ptr_filename << std::endl;
    std::cout << "rank " << rank << " col_indices_filename " << col_indices_filename << std::endl;
    std::cout << "rank " << rank << " rhs_filename " << rhs_filename << std::endl;
    std::cout << "rank " << rank << " solution_filename " << solution_filename << std::endl;

    load_binary_array<double>(data_filename, data, nnz);
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);
    load_binary_array<double>(rhs_filename, rhs, matrix_size);
    load_binary_array<double>(solution_filename, reference_solution, matrix_size);

    std::cout << "rank " << rank << " data loaded" << std::endl;

    if(rank == 0){
        if(nnz != row_ptr[matrix_size]){
            std::cout << "matrix_size " << matrix_size << std::endl;
            std::cout << "nnz " << nnz << std::endl;
            std::cout << "row_ptr[matrix_size] " << row_ptr[matrix_size] << std::endl;
            std::cout << "nnz != row_ptr[matrix_size]" << std::endl;
            exit(1);
        }
    }

    double *starting_guess_h = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess_h[i] = 0;
    }



    double times_gpu_packing[number_of_measurements];
    double times_gpu_packing_cam[number_of_measurements];

    double *test_solution_h = new double[rows_this_rank];
    test_preconditioned<dspmv::gpu_packing>(
        data,
        col_indices,
        row_ptr,
        rhs,
        starting_guess_h + row_start_index,
        test_solution_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        times_gpu_packing,
        number_of_measurements
    );

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_this_rank; ++i) {
        difference += std::sqrt( (test_solution_h[i] - reference_solution[i+row_start_index]) *
            (test_solution_h[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    // std::string path_solve_gpu_packing = get_filename(save_path, "solve_gpu_packing_"+std::to_string(matsize), size, rank);
    // std::string path_solve_gpu_packing_cam = get_filename(save_path, "solve_gpu_packing_cam_"+std::to_string(matsize), size, rank);

    // save_measurements(path_solve_gpu_packing,
    //     times_gpu_packing + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_solve_gpu_packing_cam,
    //     times_gpu_packing_cam + start_up_measurements,
    //     true_number_of_measurements, true);


    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] starting_guess_h;

    MPI_Finalize();
    return 0;
}
