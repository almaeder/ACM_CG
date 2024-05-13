#pragma once
#include "utils_cg.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "dist_objects.h"

class Preconditioner{

    public:
        virtual void  apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream,
            rocsparse_handle &default_rocsparseHandle) = 0;

};

class Preconditioner_none : public Preconditioner{
    private:
        int rows_this_rank;
    public:
        Preconditioner_none(
            Distributed_matrix &A_distributed
        );

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream,
            rocsparse_handle &default_rocsparseHandle) override;

};

class Preconditioner_jacobi : public Preconditioner{
    private:
        int rows_this_rank;
        double *diag_inv_d;
    public:
        Preconditioner_jacobi(
            Distributed_matrix &A_distributed
        );
        ~Preconditioner_jacobi();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream,
            rocsparse_handle &default_rocsparseHandle) override;

};

class Preconditioner_jacobi_split : public Preconditioner{
    private:
        int rows_this_rank;
        double *diag_inv_d;
    public:
        Preconditioner_jacobi_split(
            Distributed_matrix &A_distributed,
            Distributed_subblock &A_subblock_distributed
        );
        ~Preconditioner_jacobi_split();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream,
            rocsparse_handle &default_rocsparseHandle) override;

};

class Preconditioner_block_icholesky : public Preconditioner{
    private:
        int rows_this_rank;
        int nnz;
        int *row_ptr_d;
        int *col_ind_d;
        double *data_d;
        double *y_d;
        double *x_d;

        rocsparse_mat_descr descr_L;
        rocsparse_mat_descr descr_Lt;
        rocsparse_mat_info info;
        void* temp_buffer_d;
    public:
        Preconditioner_block_icholesky(
            Distributed_matrix &A_distributed
        );
        ~Preconditioner_block_icholesky();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream,
            rocsparse_handle &default_rocsparseHandle) override;

};