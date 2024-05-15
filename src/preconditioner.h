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
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) = 0;

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
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) override;

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
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) override;

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
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) override;

};

class Preconditioner_block_ilu : public Preconditioner{
    private:
        int rows_this_rank;
        int nnz;
        int *row_ptr_d;
        int *col_ind_d;
        double *data_d;
        double *y_d;
        double *x_d;
        hipsparseDnVecDescr_t vecY, vecX;

        hipsparseSpMatDescr_t descr_L, descr_Lt;
        hipsparseSpSVDescr_t spsvDescr_L, spsvDescr_Lt;
        csrilu02Info_t info;
        void *buffer_LLt_d, *buffer_L_d, *buffer_Lt_d;


    public:
        Preconditioner_block_ilu(
            Distributed_matrix &A_distributed
        );
        ~Preconditioner_block_ilu();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) override;

};
class Preconditioner_block_ic : public Preconditioner{
    private:
        int rows_this_rank;
        int nnz;
        int *row_ptr_d;
        int *col_ind_d;
        double *data_d;
        double *y_d;
        double *x_d;
        hipsparseDnVecDescr_t vecY, vecX;

        hipsparseSpMatDescr_t descr_L, descr_Lt;
        hipsparseSpSVDescr_t spsvDescr_L, spsvDescr_Lt;
        csric02Info_t info;
        void *buffer_LLt_d, *buffer_L_d, *buffer_Lt_d;


    public:
        Preconditioner_block_ic(
            Distributed_matrix &A_distributed
        );
        ~Preconditioner_block_ic();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipsparseDnVecDescr_t vecZ,
            hipsparseDnVecDescr_t vecR,
            hipStream_t &default_stream,
            hipsparseHandle_t &default_rocsparseHandle) override;

};