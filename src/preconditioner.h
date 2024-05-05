#pragma once
#include "utils_cg.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "dist_objects.h"

enum class Preconditioner_type { NONE, JACOBI};

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

        void apply_preconditioner(
            double *z_d,
            double *r_d,
            hipStream_t &default_stream) override;

};