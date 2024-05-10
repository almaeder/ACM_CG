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


// template <typename T>
// void save_bin_array2(T* array, int numElements, const std::string& filename) {
//     std::ofstream file(filename, std::ios::binary);
//     if (file.is_open()) {
//         file.write(reinterpret_cast<char*>(array), numElements*sizeof(T));
//         file.close();
//         std::cout << "Array data written to file: " << filename << std::endl;
//     } else {
//         std::cerr << "Unable to open the file for writing." << std::endl;
//     }
// }

void expand_row_ptr(
    int *row_ptr_d,
    int *row_ptr_compressed_d,
    int *subblock_indices_local_d,
    int matrix_size,
    int subblock_size_local
){
    int *row_ptr_h = new int[matrix_size+1];
    int *row_ptr_compressed_h = new int[subblock_size_local+1];
    int *subblock_indices_local_h = new int[subblock_size_local];

    hipMemcpy(row_ptr_compressed_h, row_ptr_compressed_d, (subblock_size_local+1)*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(subblock_indices_local_h, subblock_indices_local_d, (subblock_size_local)*sizeof(int), hipMemcpyDeviceToHost);

    for(int i = 0; i < matrix_size+1; i++){
        row_ptr_h[i] = 0;
    }
    row_ptr_h[matrix_size] = row_ptr_compressed_h[subblock_size_local];
    for(int i = 0; i < subblock_size_local; i++){
        if(i == subblock_size_local-1){
            for(int j = subblock_indices_local_h[i]+1; j < matrix_size+1; j++){
                row_ptr_h[j] = row_ptr_compressed_h[i+1];
            }
        }
        else{
            for(int j = subblock_indices_local_h[i]+1; j <  subblock_indices_local_h[i+1]+1; j++){
                row_ptr_h[j] = row_ptr_compressed_h[i+1];
            }
        }
    }
    // std::string filename = "tmp_row_ptr.bin";
    // save_bin_array2<int>(row_ptr_h, matrix_size+1, filename);

    hipMemcpy(row_ptr_d, row_ptr_h, (matrix_size+1)*sizeof(int), hipMemcpyHostToDevice);

}
void expand_col_indices(
    int *col_indices_d,
    int *col_indices_compressed_d,
    int *subblock_indices_d,
    int nnz,
    int subblock_size
){
    int *col_indices_h = new int[nnz];
    int *col_indices_compressed_h = new int[nnz];
    int *subblock_indices_h = new int[subblock_size];

    hipMemcpy(col_indices_compressed_h, col_indices_compressed_d, (nnz)*sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(subblock_indices_h, subblock_indices_d, (subblock_size)*sizeof(int), hipMemcpyDeviceToHost);

    for(int i = 0; i < nnz; i++){
        col_indices_h[i] = subblock_indices_h[col_indices_compressed_h[i]];
    }

    // std::string filename = "tmp_col_indices.bin";
    // save_bin_array2<int>(col_indices_h, nnz, filename);
    hipMemcpy(col_indices_d, col_indices_h, (nnz)*sizeof(int), hipMemcpyHostToDevice);

}

__global__ void _compression_array(
    int *compression_d,
    int *compression_indices_d,
    int number_cols_compressed
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_cols_compressed; i += blockDim.x * gridDim.x){
        compression_d[compression_indices_d[i]] = i;
    }
}

__global__ void _compress_col_indices(
    int *col_indices_d,
    int *col_indices_compressed_d,
    int *compression_d,
    int nnz
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < nnz; i += blockDim.x * gridDim.x){
        col_indices_compressed_d[i] = compression_d[col_indices_d[i]];
    }
}


void compress_col_ind(
    int *col_indices_d,
    int *col_indices_compressed_d,
    int *compression_indices_d,
    int nnz,
    int number_cols,
    int number_cols_compressed
){

    // compression[cols_per_neighbour_d[k]] = k
    // compression[col_inds_d[j]] = col_inds_d[j] - cols_per_neighbour_d[...]
    // helper array to compress
    int *compression_d;
    cudaErrchk(hipMalloc((void **)&compression_d,
        number_cols * sizeof(int)));

    int block_size = 1024;
    int num_blocks = (number_cols_compressed + block_size - 1) / block_size;
    hipLaunchKernelGGL(_compression_array, num_blocks, block_size, 0, 0,
        compression_d,
        compression_indices_d,
        number_cols_compressed);

    num_blocks = (nnz + block_size - 1) / block_size;
    hipLaunchKernelGGL(_compress_col_indices, num_blocks, block_size, 0, 0,
        col_indices_d,
        col_indices_compressed_d,
        compression_d,
        nnz);

    cudaErrchk(hipFree(compression_d));
}


__global__ void _pack(
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

void pack(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_pack, num_blocks, block_size, 0, 0, 
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

void pack(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_pack, num_blocks, block_size, 0, stream, 
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

__global__ void _unpack(
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

void unpack(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack, num_blocks, block_size, 0, 0, 
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}

void unpack(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    hipStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    hipLaunchKernelGGL(_unpack, num_blocks, block_size, 0, stream, 
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
