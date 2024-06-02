import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Generate matrix")
    seed = 10

    matrix_size = 2001

    subblock_size = 211

    matsize = 2

    save_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices/"
    rng = np.random.default_rng(seed=seed)

    all_offsets = np.arange(1, (matrix_size//16))
    chosen_diags = 20

    offsets = rng.choice(all_offsets, size=chosen_diags, replace=False)
    data = rng.uniform(size=(chosen_diags, matrix_size))

    matrix = sparse.dia_array((data, offsets), shape=(matrix_size, matrix_size), dtype=np.float64)


    subblock_all_offsets = np.arange(1, (subblock_size))
    subblock_chosen_diags = subblock_size//4

    subblock_offsets = rng.choice(subblock_all_offsets, size=subblock_chosen_diags, replace=False)
    subblock_data = rng.uniform(size=(subblock_chosen_diags, subblock_size))

    subblock_matrix = sparse.dia_array((subblock_data, subblock_offsets), shape=(subblock_size, subblock_size), dtype=np.float64)


    print("Symmetrizing matrix")
    matrix = -matrix - matrix.transpose()
    subblock_matrix = -subblock_matrix - subblock_matrix.transpose()

    subblock_indices = rng.choice(matrix_size, size=subblock_size, replace=False)
    subblock_indices = np.sort(subblock_indices)

    print("Making diagonally dominant")
    diag_sum = np.abs(np.copy(matrix.sum(axis=1)).flatten() - matrix.diagonal())
    subblock_diag_sum = np.abs(np.copy(subblock_matrix.sum(axis=1)).flatten() - subblock_matrix.diagonal())
    diag_sum += 0.0001*diag_sum

    for i in range(subblock_size):
        diag_sum[subblock_indices[i]] += subblock_diag_sum[i]

    matrix.setdiag(diag_sum)


    print("Generate total matrix")
    total_matrix = matrix.toarray()
    total_matrix[np.ix_(subblock_indices, subblock_indices)] += subblock_matrix.toarray()

    total_matrix = sparse.csr_matrix(total_matrix)
    total_matrix.eliminate_zeros()
    total_matrix.sort_indices()

    subblock_matrix = subblock_matrix.tocsr()
    subblock_matrix.eliminate_zeros()
    subblock_matrix.sort_indices()

    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    rhs = rng.uniform(size=(matrix_size,1))

    plt.spy(total_matrix.toarray())
    plt.savefig("total_matrix.png")


    print("Solving")
    solution = np.linalg.solve(total_matrix.toarray(),rhs)

    nnz_total = total_matrix.nnz
    nnz_subblock = subblock_matrix.nnz
    nnz_sparse = matrix.nnz

    assert nnz_total <= nnz_sparse + nnz_subblock
    


    print("Saving")
    with open(save_path + "X_matrix_size_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(matrix_size, dtype=np.int32).tobytes())
    with open(save_path + "X_subblock_size_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(subblock_size, dtype=np.int32).tobytes())

    with open(save_path + "X_nnz_total_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(nnz_total, dtype=np.int32).tobytes())
    with open(save_path + "X_nnz_subblock_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(nnz_subblock, dtype=np.int32).tobytes())
    with open(save_path + "X_nnz_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(nnz_sparse, dtype=np.int32).tobytes())

    with open(save_path + "X_rhs_" + str(matsize) + ".bin", "wb") as f:
        f.write(rhs.reshape((1, -1)).tobytes())
    with open(save_path + "X_solution_" + str(matsize) + ".bin", "wb") as f:
        f.write(solution.reshape((1, -1)).tobytes())

    with open(save_path + "X_subblock_indices_" + str(matsize) + ".bin", "wb") as f:
        f.write(subblock_indices.astype(np.int32).tobytes())

    with open(save_path + "X_subblock_data_" + str(matsize) + ".bin", "wb") as f:
        f.write(subblock_matrix.data.reshape((1, -1)).tobytes())
    with open(save_path + "X_subblock_col_indices_" + str(matsize) + ".bin", "wb") as f:
        indices = subblock_matrix.indices.astype(np.int32)
        f.write(indices.reshape((1, -1)).tobytes())
    with open(save_path + "X_subblock_row_ptr_" + str(matsize) + ".bin", "wb") as f:
        indptr = subblock_matrix.indptr.astype(np.int32)
        f.write(indptr.reshape((1, -1)).tobytes())

    with open(save_path + "X_data_" + str(matsize) + ".bin", "wb") as f:
        f.write(matrix.data.reshape((1, -1)).tobytes())
    with open(save_path + "X_col_indices_" + str(matsize) + ".bin", "wb") as f:
        indices = matrix.indices.astype(np.int32)
        f.write(indices.reshape((1, -1)).tobytes())
    with open(save_path + "X_row_ptr_" + str(matsize) + ".bin", "wb") as f:
        indptr = matrix.indptr.astype(np.int32)
        f.write(indptr.reshape((1, -1)).tobytes())
    
    with open(save_path + "X_total_data_" + str(matsize) + ".bin", "wb") as f:
        f.write(total_matrix.data.reshape((1, -1)).tobytes())
    with open(save_path + "X_total_col_indices_" + str(matsize) + ".bin", "wb") as f:
        indices = total_matrix.indices.astype(np.int32)
        f.write(indices.reshape((1, -1)).tobytes())
    with open(save_path + "X_total_row_ptr_" + str(matsize) + ".bin", "wb") as f:
        indptr = total_matrix.indptr.astype(np.int32)
        f.write(indptr.reshape((1, -1)).tobytes())
    