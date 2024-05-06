#include "hip/hip_runtime.h"
#include "utils_cg.h"

__global__ void _extract_diagonal_inv(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            if(col_indices[j] == i){
                diagonal_values_inv_sqrt[i] = 1/data[j];
                break;
            }
        }
    }

}

void extract_diagonal_inv(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_extract_diagonal_inv, num_blocks, block_size, 0, 0, 
        data,
        col_indices,
        row_indptr,
        diagonal_values_inv,
        matrix_size
    );
}

__global__ void _extract_diagonal(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            if(col_indices[j] == i){
                diagonal_values[i] = data[j];
                break;
            }
        }
    }

}

void extract_diagonal(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_extract_diagonal, num_blocks, block_size, 0, 0, 
        data,
        col_indices,
        row_indptr,
        diagonal_values,
        matrix_size
    );
}

__global__ void  _extract_add_subblock_diagonal(
    int *subblock_indices_d,
    int *subblock_row_ptr_d,
    int *subblock_col_indices_d,
    double *subblock_data_d,
    double *diag_inv_d,
    int subblock_rows_size,
    int displ_subblock_this_rank
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < subblock_rows_size; i += blockDim.x * gridDim.x){
        for(int j = subblock_row_ptr_d[i]; j < subblock_row_ptr_d[i+1]; j++){
            if(subblock_col_indices_d[j] == i + displ_subblock_this_rank){
                diag_inv_d[subblock_indices_d[i]] += subblock_data_d[j];
                break;
            }
        }
    }
}

void  extract_add_subblock_diagonal(
    int *subblock_indices_d,
    int *subblock_row_ptr_d,
    int *subblock_col_indices_d,
    double *subblock_data_d,
    double *diag_inv_d,
    int subblock_rows_size,
    int displ_subblock_this_rank
){
    int block_size = 1024;
    int num_blocks = (subblock_rows_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_extract_add_subblock_diagonal, num_blocks, block_size, 0, 0, 
        subblock_indices_d,
        subblock_row_ptr_d,
        subblock_col_indices_d,
        subblock_data_d,
        diag_inv_d,
        subblock_rows_size,
        displ_subblock_this_rank);

}

__global__ void _inv_inplace(
    double *data_d,
    int size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < size; i += blockDim.x * gridDim.x){
        data_d[i] = 1/data_d[i];
    }
}

void inv_inplace(
    double *data_d,
    int size
){
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_inv_inplace, num_blocks, block_size, 0, 0,
        data_d,
        size
    );
}

__global__ void _pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_of_elements; i += blockDim.x * gridDim.x){
        packed_buffer[i] = unpacked_buffer[indices[i]];
    }
}

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_pack_gpu, num_blocks, block_size, 0, 0, 
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_pack_gpu, num_blocks, block_size, 0, stream, 
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

__global__ void _unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_of_elements; i += blockDim.x * gridDim.x){
        unpacked_buffer[indices[i]] = packed_buffer[i];
    }
}

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack_gpu, num_blocks, block_size, 0, 0, 
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack_gpu, num_blocks, block_size, 0, stream, 
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}

__global__ void _unpack_add(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_of_elements; i += blockDim.x * gridDim.x){
        unpacked_buffer[indices[i]] += packed_buffer[i];
    }
}

void unpack_add(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack_add, num_blocks, block_size, 0, 0, 
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}


void unpack_add(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack_add, num_blocks, block_size, 0, stream, 
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}


__global__ void _cg_addvec(
    double * __restrict__ x,
    double beta,
    double * __restrict__ y,
    int n
)
{
    // y = x + beta * y
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x){
        y[i] = x[i] + beta * y[i];
    }
}

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(_cg_addvec, num_blocks, block_size, 0, 0, x, beta, y, n);
}

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n,
    hipStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(_cg_addvec, num_blocks, block_size, 0, stream, x, beta, y, n);
}

__global__ void _fused_daxpy(
    double alpha1,
    double alpha2,
    double * __restrict__ x1,
    double * __restrict__ x2,
    double * __restrict__ y1,
    double * __restrict__ y2,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x){
        y1[i] = y1[i] + alpha1 * x1[i];
        y2[i] = y2[i] + alpha2 * x2[i];
    }
}

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(_fused_daxpy, num_blocks, block_size, 0, 0, 
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    hipStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(_fused_daxpy, num_blocks, block_size, 0, stream, 
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

__global__ void _fused_daxpy2(
    double alpha1,
    double alpha2,
    double * __restrict__ x1,
    double * __restrict__ x2,
    double * __restrict__ y1,
    double * __restrict__ y2,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx - blockDim.x * gridDim.x / 2;
    if(idx < n){
        y1[idx] = y1[idx] + alpha1 * x1[idx];
    }
    else if(idx2 >= 0 && idx2 < n){
        y2[idx2] = y2[idx2] + alpha2 * x2[idx2];
    }

}

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    num_blocks *= 2;
    hipLaunchKernelGGL(_fused_daxpy2, num_blocks, block_size, 0, 0, 
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    hipStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    num_blocks *= 2;
    hipLaunchKernelGGL(_fused_daxpy2, num_blocks, block_size, 0, stream, 
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

__global__ void _elementwise_vector_vector(
    double * __restrict__ array1,
    double * __restrict__ array2,
    double * __restrict__ result,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < size; i += blockDim.x * gridDim.x){
        result[i] = array1[i] * array2[i];
    }

}

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size
)
{
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_elementwise_vector_vector, num_blocks, block_size, 0, 0, 
        array1,
        array2,
        result,
        size
    );
}

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size,
    hipStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    hipLaunchKernelGGL(_elementwise_vector_vector, num_blocks, block_size, 0, stream, 
        array1,
        array2,
        result,
        size
    );
}
