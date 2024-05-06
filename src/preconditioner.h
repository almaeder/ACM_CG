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
        hipStream_t &default_stream) = 0;

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
            hipStream_t &default_stream) override;

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
            hipStream_t &default_stream) override;

};

class Preconditioner_jacobi_split : public Preconditioner{
    private:
        int rows_this_rank;
        double *diag_inv_d;
    public:
        Preconditioner_jacobi_split(
            Distributed_matrix &A_distributed,
            int subblock_rows_size,
            int *subblock_indices_d,
            int *subblock_row_ptr_d,
            int *subblock_col_indices_d,
            double *subblock_data_d,
            int displ_subblock_this_rank
        );
        ~Preconditioner_jacobi_split();

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream) override;

};