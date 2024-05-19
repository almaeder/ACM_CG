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
        std::cout << "nnz tot " << nnz_total << std::endl;
        std::cout << "nnz blue " << nnz_sparse << std::endl;
        std::cout << "nnz green " << nnz_subblock << std::endl;
    }

    if(nnz_subblock + nnz_sparse < nnz_total){
        std::cout << "Impossible nnz" << std::endl;
        std::cout << nnz_subblock << " " << nnz_sparse << " " << nnz_total << std::endl; 
        exit(1);
    }

    int max_iterations = 10000;
    double relative_tolerance =  1e-8;

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

        // correct total dat since there is a problem:
        double *diag_values = new double[matrix_size];
        for(int i = 0; i < matrix_size; i++){
            diag_values[i] = 0; 
        }
        for(int i = 0; i < matrix_size; i++){
            for(int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; j++){
                if(i == col_indices_sparse[j]){
                    diag_values[i] = data_sparse[j];     
                }
            }
        }
        for(int i = 0; i < subblock_size; i++){
            for(int j = row_ptr_subblock[i]; j < row_ptr_subblock[i+1]; j++){
                if(i == col_indices_subblock[j]){
                    diag_values[subblock_indices[i]] += data_subblock[j];     
                }
            }
        }
        for(int i = 0; i < matrix_size; i++){
            for(int j = row_ptr_tot[i]; j < row_ptr_tot[i+1]; j++){
                if(i == col_indices_tot[j]){
                    data_tot[j] = diag_values[i];     
                }
            }
        }
        delete[] diag_values;

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

    if(rank == 0 && false){
        // check matrices the same
        double *dense_tot = new double[matrix_size * matrix_size];
        double *dense_split = new double[matrix_size * matrix_size];
        // double *dense_split2 = new double[matrix_size * matrix_size];

        // std::string row_ptr_subblock_filename2 = "tmp_row_ptr.bin";
        // std::string col_indices_subblock_filename2 = "tmp_col_indices.bin";
        // int *row_ptr_subblock2 = new int[matrix_size+1];
        // int *col_indices_subblock2 = new int[nnz_subblock];
        // load_binary_array<int>(row_ptr_subblock_filename2, row_ptr_subblock2, matrix_size+1);
        // load_binary_array<int>(col_indices_subblock_filename2, col_indices_subblock2, nnz_subblock);

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
        // for (int i = 0; i < matrix_size; ++i) {
        //     for (int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; ++j) {
        //         dense_split2[i * matrix_size + col_indices_sparse[j]] = data_sparse[j];
        //     }
        // }
        // for (int i = 0; i < matrix_size; ++i) {
        //     for (int j = row_ptr_subblock2[i]; j < row_ptr_subblock2[i+1]; ++j) {
        //         dense_split2[i * matrix_size + col_indices_subblock2[j]] += data_subblock[j];
        //     }
        // }


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
        delete[] dense_tot;
        delete[] dense_split;
        // delete[] dense_split2;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int start_up_measurements = 0;
    int true_number_of_measurements = 3;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    
    double times_not_split_overlap[number_of_measurements];
    double times_not_split_singelkernel[number_of_measurements];
    double times_overlap_compressed1[number_of_measurements];
    double times_overlap_compressed2[number_of_measurements];
    double times_singlekernel_compressed1[number_of_measurements];
    double times_singlekernel_compressed2[number_of_measurements];
    double times_singlekernel_compressed3[number_of_measurements];

    double *test_solution1_h = new double[matrix_size];
    double *test_solution2_h = new double[matrix_size];
    double *test_solution3_h = new double[matrix_size];
    double *test_solution4_h = new double[matrix_size];
    double *test_solution5_h = new double[matrix_size];
    double *test_solution6_h = new double[matrix_size];
    double *test_solution7_h = new double[matrix_size];

    // test_preconditioned<dspmv::manual_packing_overlap_compressed, Preconditioner_jacobi>(
    //     data_tot,
    //     col_indices_tot,
    //     row_ptr_tot,
    //     rhs,
    //     starting_guess_h,
    //     test_solution1_h,
    //     matrix_size,
    //     relative_tolerance,
    //     max_iterations,
    //     MPI_COMM_WORLD,
    //     times_not_split_overlap,
    //     number_of_measurements
    // );

    // if(rank == 0){
    //     relative_error(matrix_size, test_solution1_h, reference_solution);
    // }

    // test_preconditioned<dspmv::manual_packing_singlekernel_compressed, Preconditioner_jacobi>(
    //     data_tot,
    //     col_indices_tot,
    //     row_ptr_tot,
    //     rhs,
    //     starting_guess_h,
    //     test_solution2_h,
    //     matrix_size,
    //     relative_tolerance,
    //     max_iterations,
    //     MPI_COMM_WORLD,
    //     times_not_split_singelkernel,
    //     number_of_measurements
    // );

    // if(rank == 0){
    //     relative_error(matrix_size, test_solution2_h, reference_solution);
    // }

    // test_preconditioned_split_sparse<dspmv_split_sparse::manual_packing_overlap_compressed1>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         starting_guess_h,
    //         test_solution3_h,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         times_overlap_compressed1,
    //         number_of_measurements
    // );
    // if(rank == 0){
    //     relative_error(matrix_size, test_solution3_h, reference_solution);
    // }

    // test_preconditioned_split_sparse<dspmv_split_sparse::manual_packing_overlap_compressed2>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         starting_guess_h,
    //         test_solution4_h,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         times_overlap_compressed2,
    //         number_of_measurements
    // );
    // if(rank == 0){
    //     relative_error(matrix_size, test_solution4_h, reference_solution);
    // }

    // test_preconditioned_split_sparse<dspmv_split_sparse::manual_packing_singlekernel_compressed1>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         starting_guess_h,
    //         test_solution5_h,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         times_singlekernel_compressed1,
    //         number_of_measurements
    // );
    // if(rank == 0){
    //     relative_error(matrix_size, test_solution5_h, reference_solution);
    // }

    // test_preconditioned_split_sparse<dspmv_split_sparse::manual_packing_singlekernel_compressed2>(
    //         data_sparse,
    //         col_indices_sparse,
    //         row_ptr_sparse,
    //         subblock_indices,
    //         data_subblock,
    //         col_indices_subblock,
    //         row_ptr_subblock,
    //         subblock_size,
    //         rhs,
    //         starting_guess_h,
    //         test_solution6_h,
    //         matrix_size,
    //         relative_tolerance,
    //         max_iterations,
    //         MPI_COMM_WORLD,
    //         times_singlekernel_compressed2,
    //         number_of_measurements
    // );
    // if(rank == 0){
    //     relative_error(matrix_size, test_solution6_h, reference_solution);
    // }

    test_preconditioned_split_sparse<dspmv_split_sparse::manual_packing_singlekernel_compressed3>(
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
            test_solution7_h,
            matrix_size,
            relative_tolerance,
            max_iterations,
            MPI_COMM_WORLD,
            times_singlekernel_compressed3,
            number_of_measurements
    );
    if(rank == 0){
        relative_error(matrix_size, test_solution7_h, reference_solution);
    }

    // std::string path_not_split_overlap = get_filename(save_path, "X_not_split_overlap", size, rank);
    // std::string path_not_split_singelkernel = get_filename(save_path, "X_not_split_singelkernel", size, rank);
    // std::string path_overlap_compressed1 = get_filename(save_path, "X_overlap_compressed1", size, rank);
    // std::string path_overlap_compressed2 = get_filename(save_path, "X_overlap_compressed2", size, rank);
    // std::string path_singlekernel_compressed1 = get_filename(save_path, "X_singlekernel_compressed1", size, rank);
    // std::string path_singlekernel_compressed2 = get_filename(save_path, "X_singlekernel_compressed2", size, rank);

    // save_measurements(path_not_split_overlap,
    //     times_not_split_overlap + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_not_split_singelkernel,
    //     times_not_split_singelkernel + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_overlap_compressed1,
    //     times_overlap_compressed1 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_overlap_compressed2,
    //     times_overlap_compressed2 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_singlekernel_compressed1,
    //     times_singlekernel_compressed1 + start_up_measurements,
    //     true_number_of_measurements, true);
    // save_measurements(path_singlekernel_compressed2,
    //     times_singlekernel_compressed2 + start_up_measurements,
    //     true_number_of_measurements, true);

    delete[] data_sparse;
    delete[] row_ptr_sparse;
    delete[] col_indices_sparse;
    delete[] data_tot;
    delete[] row_ptr_tot;
    delete[] col_indices_tot;
    delete[] reference_solution;
    delete[] rhs;
    delete[] starting_guess_h;
    delete[] subblock_indices;
    delete[] data_subblock;
    delete[] row_ptr_subblock;
    delete[] col_indices_subblock;

    MPI_Finalize();
    return 0;
}
