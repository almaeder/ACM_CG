#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <hip/hip_runtime.h>
#include "nondist_wrapper.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char* slurm_localid = getenv("SLURM_LOCALID");
    int localid = -1;
    if (slurm_localid != nullptr) {
        localid = atoi(slurm_localid);
        std::cout << "Rank " << rank << " has SLURM_LOCALID " << localid << std::endl;
    } else {
        std::cerr << "Rank " << rank << " cannot access SLURM_LOCALID" << std::endl;
        exit(1);
    }

    char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        std::cout << "Rank " << rank << " ROCR_VISIBLE_DEVICES: " << rocr_visible_devices << std::endl;
    } else {
        std::cerr << "Rank " << rank << " ROCR_VISIBLE_DEVICES not set" << std::endl;
        exit(1);
    }

    hipError_t set_device_error = hipSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    int matsize = 401;
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
    int true_number_of_measurements = 2;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;

    int max_iterations = 5000;
    double relative_tolerance = 1e-9;

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];

    if(rank == 0){
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


        std::cout << "Loading data" << std::endl;
        std::cout << "data_filename " << data_filename << std::endl;
        std::cout << "row_ptr_filename " << row_ptr_filename << std::endl;
        std::cout << "col_indices_filename " << col_indices_filename << std::endl;
        std::cout << "rhs_filename " << rhs_filename << std::endl;
        std::cout << "solution_filename " << solution_filename << std::endl;

        load_binary_array<double>(data_filename, data, nnz);
        load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
        load_binary_array<int>(col_indices_filename, col_indices, nnz);
        load_binary_array<double>(rhs_filename, rhs, matrix_size);
        load_binary_array<double>(solution_filename, reference_solution, matrix_size);

        std::cout << "data loaded" << std::endl;
    }
    MPI_Bcast(data, nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr, matrix_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices, nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rhs, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(reference_solution, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);


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

    double *test_solution1_h = new double[matrix_size];
    test_preconditioned<dspmv::alltoall_cam>(
        data,
        col_indices,
        row_ptr,
        rhs,
        starting_guess_h,
        test_solution1_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        times_gpu_packing,
        number_of_measurements
    );

    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution1_h[i] - reference_solution[i]) *
                (test_solution1_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    double *test_solution2_h = new double[matrix_size];
    test_preconditioned<dspmv::pointpoint_singlekernel_cam>(
        data,
        col_indices,
        row_ptr,
        rhs,
        starting_guess_h,
        test_solution2_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        times_gpu_packing,
        number_of_measurements
    );

    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution2_h[i] - reference_solution[i]) *
                (test_solution2_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    double *test_solution3_h = new double[matrix_size];
    test_preconditioned<dspmv::manual_packing_cam>(
        data,
        col_indices,
        row_ptr,
        rhs,
        starting_guess_h,
        test_solution3_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        times_gpu_packing,
        number_of_measurements
    );

    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution3_h[i] - reference_solution[i]) *
                (test_solution3_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    double *test_solution4_h = new double[matrix_size];
    test_preconditioned<dspmv::manual_packing_cam2>(
        data,
        col_indices,
        row_ptr,
        rhs,
        starting_guess_h,
        test_solution4_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        times_gpu_packing,
        number_of_measurements
    );

    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution4_h[i] - reference_solution[i]) *
                (test_solution4_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
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
