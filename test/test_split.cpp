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
    int localid = atoi(slurm_localid);
    int device_id = localid; 
    std::cout << "rank " << rank << " device_id " << device_id << std::endl;

    hipError_t set_device_error = hipSetDevice(localid);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    int matsize = 2;
    std::string data_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices/";
    std::string save_path ="/scratch/project_465000929/maederal/ACM_Poster/results/";

    int matrix_size, subblock_size, nnz_subblock, nnz_sparse, nnz_total;

    load_binary_array<int>(data_path + "X_matrix_size"+ "_"+ std::to_string(matsize)+".bin", &matrix_size, 1);
    load_binary_array<int>(data_path + "X_subblock_size"+ "_"+ std::to_string(matsize)+".bin", &subblock_size, 1);
    load_binary_array<int>(data_path + "X_subblock_nnz"+ "_"+ std::to_string(matsize)+".bin", &nnz_subblock, 1);
    load_binary_array<int>(data_path + "X_nnz"+ "_"+ std::to_string(matsize)+".bin", &nnz_sparse, 1);
    load_binary_array<int>(data_path + "X_total_nnz"+ "_"+ std::to_string(matsize)+".bin", &nnz_total, 1);

    std::cout << "rank " << rank << ", loaded matrix parameters" << std::endl;

    if(rank == 0){
        std::cout << "matrix_size " << matrix_size << std::endl;
        std::cout << "subblock_size " << subblock_size << std::endl;
    }

    if(nnz_subblock + nnz_sparse < nnz_total){
        std::cout << "Impossible nnz" << std::endl;
        exit(1);
    }



    int max_iterations = 10000;
    double relative_tolerance = 1e-18;

    int counts[size];
    int displacements[size];
    int rows_this_rank;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_this_rank = counts[rank];
    int row_end_index = row_start_index + rows_this_rank;

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " row_end_index " << row_end_index << std::endl;
    std::cout << "rank " << rank << " rows_this_rank " << rows_this_rank << std::endl;

    double *data_sparse = new double[nnz_sparse];
    int *row_ptr_sparse = new int[matrix_size+1];
    int *col_indices_sparse = new int[nnz_sparse];

    double *data_tot = new double[nnz_total];
    int *row_ptr_tot = new int[matrix_size+1];
    int *col_indices_tot = new int[nnz_total];

    double *reference_solution = new double[matrix_size];
    double *rhs = new double[matrix_size];
    double *starting_guess_h = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess_h[i] = 0.0;
    }

    int *subblock_indices = new int[subblock_size];
    double *data_subblock = new double[nnz_subblock];
    int *row_ptr_subblock = new int[subblock_size+1];
    int *col_indices_subblock = new int[nnz_subblock];
    
    std::string data_sparse_filename = data_path + "X_data_"+ std::to_string(matsize) +".bin";
    std::string row_ptr_sparse_filename = data_path + "X_row_ptr_"+ std::to_string(matsize) +".bin";
    std::string col_indices_sparse_filename = data_path + "X_col_indices_"+ std::to_string(matsize) +".bin";
    std::string data_tot_filename = data_path + "X_total_data_"+ std::to_string(matsize) +".bin";
    std::string row_ptr_tot_filename = data_path + "X_total_row_ptr_"+ std::to_string(matsize) +".bin";
    std::string col_indices_tot_filename = data_path + "X_total_col_indices_"+ std::to_string(matsize) +".bin";
    std::string rhs_filename = data_path + "X_rhs_"+std::to_string(matsize)+".bin";
    std::string solution_filename = data_path + "X_solution_"+std::to_string(matsize)+".bin";

    std::string subblock_indices_filename = data_path + "X_subblock_indices_"+std::to_string(matsize)+".bin";
    std::string data_subblock_filename = data_path + "X_subblock_data_"+std::to_string(matsize)+".bin";
    std::string row_ptr_subblock_filename = data_path + "X_subblock_row_ptr_"+std::to_string(matsize)+".bin";
    std::string col_indices_subblock_filename = data_path + "X_subblock_col_indices_"+std::to_string(matsize)+".bin";
    if(rank == 0){
        load_binary_array<double>(data_sparse_filename, data_sparse, nnz_sparse);
        load_binary_array<int>(row_ptr_sparse_filename, row_ptr_sparse, matrix_size+1);
        load_binary_array<int>(col_indices_sparse_filename, col_indices_sparse, nnz_sparse);

        load_binary_array<double>(data_tot_filename, data_tot, nnz_total);
        load_binary_array<int>(row_ptr_tot_filename, row_ptr_tot, matrix_size+1);
        load_binary_array<int>(col_indices_tot_filename, col_indices_tot, nnz_total);

        load_binary_array<double>(rhs_filename, rhs, matrix_size);
        load_binary_array<double>(solution_filename, reference_solution, matrix_size);

        load_binary_array<int>(subblock_indices_filename, subblock_indices, subblock_size);
        load_binary_array<double>(data_subblock_filename, data_subblock, nnz_subblock);
        load_binary_array<int>(row_ptr_subblock_filename, row_ptr_subblock, subblock_size+1);
        load_binary_array<int>(col_indices_subblock_filename, col_indices_subblock, nnz_subblock);
    }
    // broadcast data
    std::cout << "broadcasting data" << std::endl;
    MPI_Bcast(data_sparse, nnz_sparse, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_sparse, matrix_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_sparse, nnz_sparse, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(data_tot, nnz_total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_tot, matrix_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_tot, nnz_total, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(rhs, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(subblock_indices, subblock_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(data_subblock, nnz_subblock, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_ptr_subblock, subblock_size+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_indices_subblock, nnz_subblock, MPI_INT, 0, MPI_COMM_WORLD);

    if(row_ptr_subblock[subblock_size] != nnz_subblock){
        std::cout << "rank " << rank << " row_ptr_subblock[subblock_size] " << row_ptr_subblock[subblock_size] << " nnz_subblock " << nnz_subblock << std::endl;
        exit(1);
    }
    if(row_ptr_tot[matrix_size] != nnz_total){
        std::cout << "rank " << rank << " row_ptr_tot[matrix_size] " << row_ptr_tot[matrix_size] << " nnz_total " << nnz_total << std::endl;
        exit(1);
    }
    if(row_ptr_sparse[matrix_size] != nnz_sparse){
        std::cout << "rank " << rank << " row_ptr_sparse[matrix_size] " << row_ptr_sparse[matrix_size] << " nnz_sparse " << nnz_sparse << std::endl;
        exit(1);
    }


    std::cout << "loaded data" << std::endl;

    if(rank == 0){
        // check matrices the same
        double *dense_tot = new double[matrix_size * matrix_size];
        double *dense_split = new double[matrix_size * matrix_size];
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                dense_tot[i * matrix_size + j] = 0.0;
                dense_split[i * matrix_size + j] = 0.0;
            }
        }
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = row_ptr_tot[i]; j < row_ptr_tot[i+1]; ++j) {
                dense_tot[i * matrix_size + col_indices_tot[j]] = data_tot[j];
            }
        }
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; ++j) {
                dense_split[i * matrix_size + col_indices_sparse[j]] = data_sparse[j];
            }
        }
        for (int i = 0; i < subblock_size; ++i) {
            for (int j = row_ptr_subblock[i]; j < row_ptr_subblock[i+1]; ++j) {
                dense_split[subblock_indices[i] * matrix_size + subblock_indices[col_indices_subblock[j]]] += data_subblock[j];
            }
        }

        double sum_matrix = 0.0;
        double diff_matrix = 0.0;
        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                sum_matrix += std::abs(dense_tot[i * matrix_size + j]) * std::abs(dense_tot[i * matrix_size + j]);
                diff_matrix += std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]) *
                    std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]);
            }
        }
        std::cout << "relative between matrices " << std::sqrt(diff_matrix / sum_matrix) << std::endl;

        for (int i = 0; i < matrix_size; ++i) {
            for (int j = 0; j < matrix_size; ++j) {
                if(std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]) > 1e-10){
                    std::cout << "rank " << rank << " i " << i << " j " << j << " dense_tot " << dense_tot[i * matrix_size + j]
                        << " dense_split " << dense_split[i * matrix_size + j] << std::endl;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int start_up_measurements = 0;
    int true_number_of_measurements = 1;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    double time_tot[number_of_measurements];
    double time_split_sparse1[number_of_measurements];
    double time_split_sparse2[number_of_measurements];
    double time_split_sparse3[number_of_measurements];


    double *test_solution_tot_h = new double[matrix_size];

    test_preconditioned<dspmv::gpu_packing>(
        data_tot,
        col_indices_tot,
        row_ptr_tot,
        rhs,
        starting_guess_h,
        test_solution_tot_h,
        matrix_size,
        relative_tolerance,
        max_iterations,
        MPI_COMM_WORLD,
        time_tot,
        number_of_measurements
    );

    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution_tot_h[i] - reference_solution[i]) *
                (test_solution_tot_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
        std::cout << "difference/sum_ref tot " << difference/sum_ref << std::endl;
    }

    double *test_solution_split1_h = new double[matrix_size];

    test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse1>(
            data_sparse,
            col_indices_sparse,
            row_ptr_sparse,
            subblock_indices,
            data_subblock,
            col_indices_subblock,
            row_ptr_subblock,
            subblock_size,
            rhs,
            starting_guess_h,
            test_solution_split1_h,
            matrix_size,
            relative_tolerance,
            max_iterations,
            MPI_COMM_WORLD,
            time_split_sparse1,
            number_of_measurements
    );


    if(rank == 0){
        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < matrix_size; ++i) {
            difference += std::sqrt( (test_solution_split1_h[i] - reference_solution[i]) *
                (test_solution_split1_h[i] - reference_solution[i]) );
            sum_ref += std::sqrt( (reference_solution[i]) * (reference_solution[i]) );
        }
        std::cout << "difference/sum_ref split1 " << difference/sum_ref << std::endl;
    }
    
    // // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse2>(
    // //         data_sparse,
    // //         col_indices_sparse,
    // //         row_ptr_sparse,
    // //         subblock_indices,
    // //         data_subblock,
    // //         col_indices_subblock,
    // //         row_ptr_subblock,
    // //         subblock_size,
    // //         rhs,
    // //         test_solution_split1_h,
    // //         starting_guess_h,
    // //         matrix_size,
    // //         relative_tolerance,
    // //         max_iterations,
    // //         MPI_COMM_WORLD,
    // //         time_split_sparse2,
    // //         number_of_measurements
    // // );


    // // sum = 0.0;
    // // diff_split = 0.0;
    // // #pragma omp parallel for reduction(+:sum, diff_split)
    // // for (int i = 0; i < matrix_size; ++i) {
    // //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    // //     diff_split += std::abs(reference_solution[i] - test_solution_split1_h[i]) * std::abs(reference_solution[i] - test_solution_split1_h[i]);
    // // }
    // // if(rank == 0){
    // //     std::cout << " relative error split sparse2 " << std::sqrt(diff_split / sum) << std::endl; 
    // // }
    

    // // test_preconditioned_split_sparse<dspmv_split_sparse::spmm_split_sparse3>(
    // //         data_sparse,
    // //         col_indices_sparse,
    // //         row_ptr_sparse,
    // //         subblock_indices,
    // //         data_subblock,
    // //         col_indices_subblock,
    // //         row_ptr_subblock,
    // //         subblock_size,
    // //         rhs,
    // //         test_solution_split1_h,
    // //         starting_guess_h,
    // //         matrix_size,
    // //         relative_tolerance,
    // //         max_iterations,
    // //         MPI_COMM_WORLD,
    // //         time_split_sparse3,
    // //         number_of_measurements
    // // );


    // // sum = 0.0;
    // // diff_split = 0.0;
    // // #pragma omp parallel for reduction(+:sum, diff_split)
    // // for (int i = 0; i < matrix_size; ++i) {
    // //     sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
    // //     diff_split += std::abs(reference_solution[i] - test_solution_split1_h[i]) * std::abs(reference_solution[i] - test_solution_split1_h[i]);
    // // }
    // // if(rank == 0){
    // //     std::cout << " relative error split sparse3 " << std::sqrt(diff_split / sum) << std::endl; 
    // // }
    


    // // // rank zero should print average time
    // // double average_time_tot = 0;
    // // double average_time_split1 = 0;
    // // double average_time_split2 = 0;
    // // double average_time_split3 = 0;
    // // double average_time_split4 = 0;
    // // double average_time_split5 = 0;
    // // double average_time_split6 = 0;
    // // double average_time_split_sparse1 = 0;
    // // double average_time_split_sparse2 = 0;
    // // double average_time_split_sparse3 = 0;
    // // double average_time_split_sparse4 = 0;
    // // double average_time_split_sparse5 = 0;
    // // for(int i = 1; i < number_of_measurements; i++){
    // //     average_time_tot += time_tot[i];
    // //     average_time_split1 += time_split1[i];
    // //     average_time_split2 += time_split2[i];
    // //     average_time_split3 += time_split3[i];
    // //     average_time_split4 += time_split4[i];
    // //     average_time_split5 += time_split5[i];
    // //     average_time_split6 += time_split6[i];
    // //     average_time_split_sparse1 += time_split_sparse1[i];
    // //     average_time_split_sparse2 += time_split_sparse2[i];
    // //     average_time_split_sparse3 += time_split_sparse3[i];
    // //     average_time_split_sparse4 += time_split_sparse4[i];
    // //     average_time_split_sparse5 += time_split_sparse5[i];
    // // }
    // // // if(rank == 0){
    // // //     std::cout << "average time tot " << average_time_tot / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split1 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split2 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split3 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split4 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split5 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split " << average_time_split6 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split sparse " << average_time_split_sparse1 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split sparse " << average_time_split_sparse2 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split sparse " << average_time_split_sparse3 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split sparse " << average_time_split_sparse4 / (number_of_measurements-1) << std::endl;
    // // //     std::cout << "average time split sparse " << average_time_split_sparse5 / (number_of_measurements-1) << std::endl;
    // // // }

    // // hipFree(diag_inv_d);
    // // delete[] diag_inv_h;
    // // delete[] data_sparse;
    // // delete[] row_ptr_sparse;
    // // delete[] col_indices_sparse;
    // // delete[] data_tot;
    // // delete[] row_ptr_tot;
    // // delete[] col_indices_tot;
    // // delete[] reference_solution;
    // // delete[] rhs;
    // // delete[] starting_guess_h;
    // // delete[] subblock_indices;
    // // delete[] dense_subblock_data;
    // // delete[] diag;
    // // // delete[] dense_tot;
    // // // delete[] dense_split;
    // // std::string save_path = "/scratch/project_465000929/maederal/measurement_100_split/";
    // // std::string path_solve_tot = get_filename(save_path, "solve_tot", 0, size, rank);
    // // std::string path_solve_split1 = get_filename(save_path, "solve_split1", 0, size, rank);
    // // std::string path_solve_split2 = get_filename(save_path, "solve_split2", 0, size, rank);
    // // std::string path_solve_split3 = get_filename(save_path, "solve_split3", 0, size, rank);
    // // std::string path_solve_split4 = get_filename(save_path, "solve_split4", 0, size, rank);
    // // std::string path_solve_split5 = get_filename(save_path, "solve_split5", 0, size, rank);
    // // std::string path_solve_split6 = get_filename(save_path, "solve_split6", 0, size, rank);
    // // std::string path_solve_split_sparse1 = get_filename(save_path, "solve_split_sparse1", 0, size, rank);
    // // std::string path_solve_split_sparse2 = get_filename(save_path, "solve_split_sparse2", 0, size, rank);
    // // std::string path_solve_split_sparse3 = get_filename(save_path, "solve_split_sparse3", 0, size, rank);
    // // std::string path_solve_split_sparse4 = get_filename(save_path, "solve_split_sparse4", 0, size, rank);
    // // std::string path_solve_split_sparse5 = get_filename(save_path, "solve_split_sparse5", 0, size, rank);

    // // save_measurements(path_solve_tot,
    // //     time_tot + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split1,
    // //     time_split1 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split2,
    // //     time_split2 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split3,
    // //     time_split3 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split4,
    // //     time_split4 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split5,
    // //     time_split5 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split6,
    // //     time_split6 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split_sparse1,
    // //     time_split_sparse1 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split_sparse2,
    // //     time_split_sparse2 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split_sparse3,
    // //     time_split_sparse3 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split_sparse4,
    // //     time_split_sparse4 + start_up_measurements,
    // //     true_number_of_measurements, true);
    // // save_measurements(path_solve_split_sparse5,
    // //     time_split_sparse5 + start_up_measurements,
    // //     true_number_of_measurements, true);

    MPI_Finalize();
    return 0;
}
