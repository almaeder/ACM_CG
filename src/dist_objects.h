#pragma once
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <iostream>
#include "cudaerrchk.h"
#include <unistd.h>
#include <rocsparse.h>
#include "utils_cg.h"

class Distributed_vector{
    public:
        int matrix_size;
        int rows_this_rank;
        int size;
        int rank;
        int *counts;
        int *displacements;
        int number_of_neighbours;
        int *neighbours;
        MPI_Comm comm;

        double **vec_h;
        double **vec_d;
        rocsparse_dnvec_descr *descriptors;
        double *tot_vec_h;
        double *tot_vec_d;
        rocsparse_dnvec_descr descriptor;

    Distributed_vector(
        int matrix_size,
        int *counts,
        int *displacements,
        int number_of_neighbours,
        int *neighbours,
        MPI_Comm comm);
    ~Distributed_vector();

};

class Distributed_subblock{
    public:
        int size;
        int rank;
        int *counts;
        int *displacements;    
        MPI_Comm comm;

        int matrix_size;
        int *subblock_indices_d;
        int *subblock_indices_local_d;
        int subblock_size;

        int *counts_subblock;
        int *displacements_subblock;

        int nnz;
        double *data_d;
        int *row_uncompressed_d;
        int *row_ptr_compressed_d;
        int *col_indices_uncompressed_d;
        int *col_indices_compressed_d;
        
        rocsparse_spmat_descr descriptor_compressed;
        size_t buffersize_compressed;
        double *buffer_compressed_d;

        rocsparse_spmat_descr descriptor_uncompressed;
        size_t buffersize_uncompressed;
        double *buffer_uncompressed_d;

        rocsparse_spmv_alg algo;

        hipEvent_t *events_recv;
        hipStream_t *streams_recv;
        MPI_Request *send_requests;
        MPI_Request *recv_requests;

        Distributed_subblock(
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
            MPI_Comm comm);

        ~Distributed_subblock();


    private:


};

// assumes that the matrix is symmetric
// does a 1D decomposition over the rows
class Distributed_matrix{
    public:
        int matrix_size;
        int rows_this_rank;
        int nnz;
    
        int size;
        int rank;
        int *counts;
        int *displacements;    
        MPI_Comm comm;

        // includes itself
        int number_of_neighbours;
        // true or false if neighbour
        bool *neighbours_flag;
        // list of neighbour indices 
        // starting from own rank
        int *neighbours;

        // by default, we assume that the matrix is stored in CSR format
        // first matrix is own piece
        double **data_h;
        int **col_indices_h;
        int **row_ptr_h;

        // Data types for cuSPARSE
        size_t *buffers_size;
        double **buffers_d;
        double **datas_d;
        int **col_inds_d;
        int **row_ptrs_d;
        rocsparse_spmat_descr *descriptors;
        rocsparse_spmv_alg *algos_generic;

        size_t buffer_size;
        double *buffer_d;
        double *data_d;
        int *col_ind_d;
        int *row_ptr_d;
        rocsparse_spmat_descr descriptor;
        rocsparse_spmv_alg algo_generic;

        // Data types for MPI
        // assumes symmetric matrix
        
        // number of non-zeros per neighbour
        int *nnz_per_neighbour;
        // number of non-zeros columns per neighbour
        int *nnz_cols_per_neighbour;
        // number of non-zeros rows per neighbour
        int *nnz_rows_per_neighbour;

        // indices to fetch from neighbours
        int **cols_per_neighbour_h;
        int **cols_per_neighbour_d;
        // indices to send to neighbours
        int **rows_per_neighbour_h;
        int **rows_per_neighbour_d;
        
        // send and recv buffers
        double **send_buffer_h;
        double **recv_buffer_h;        
        double **send_buffer_d;
        double **recv_buffer_d;

        // MPI data types and requests
        MPI_Datatype *send_types;
        MPI_Datatype *recv_types;
        MPI_Request *send_requests;
        MPI_Request *recv_requests;
        // MPI streams and events
        hipStream_t *streams_recv;
        hipStream_t *streams_send;
        hipEvent_t *events_recv;
        hipEvent_t *events_send;

        hipEvent_t event_default_finished;

        // initialize cuda
        hipStream_t default_stream;
        hipblasHandle_t default_cublasHandle;
        hipsparseHandle_t default_cusparseHandle;
        rocsparse_handle default_rocsparseHandle;
        
        double *Ap_local_d;
        rocsparse_dnvec_descr vecAp_local;
        double *z_local_d;

    // construct the distributed matrix
    // input is the whol count[rank] * matrix size
    // csr part of the matrix
    Distributed_matrix(
        int matrix_size,
        int nnz,
        int *counts,
        int *displacements,
        int *col_ind_in_h,
        int *row_ptr_in_h,
        double *data_in_h,
        rocsparse_spmv_alg *algos,
        MPI_Comm comm);

    // construct the distributed matrix
    // input is correctly split
    // data is not set
    Distributed_matrix(
        int matrix_size,
        int *counts_in,
        int *displacements_in,
        int number_of_neighbours,
        int *neighbours_in,
        int **col_indices_in_d,
        int **row_ptr_in_d,
        int *nnz_per_neighbour_in,
        rocsparse_spmv_alg *algos_neighbours,
        MPI_Comm comm);


    ~Distributed_matrix();

    private:
        void find_neighbours(
            int *col_ind_in_h,
            int *row_ptr_in_h
        );

        void construct_neighbours_list(
        );

        void construct_nnz_per_neighbour(
            int *col_ind_in_h,
            int *row_ptr_in_h
        );

        void split_csr(
            int *col_ind_in_h,
            int *row_ptr_in_h,
            double *data_in_h
        );

        
        void construct_nnz_cols_per_neighbour();

        void construct_nnz_rows_per_neighbour();
        

        void construct_rows_per_neighbour();

        void construct_cols_per_neighbour(); 

        void check_sorted();

        void construct_mpi_data_types();

        void create_events_streams();

        void create_host_memory();

        void create_device_memory();

        void prepare_spmv(rocsparse_spmv_alg *algos_neighbours);

        void create_cg_overhead();

        void destroy_cg_overhead();

};