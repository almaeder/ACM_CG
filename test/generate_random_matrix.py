import numpy as np
from scipy import sparse

if __name__ == "__main__":
    seed = 10

    matrix_size = 2001
    matsize = 2

    save_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices/"
    rng = np.random.default_rng(seed=seed)

    all_offsets = np.arange(1, (matrix_size//16))
    chosen_diags = 20

    offsets = rng.choice(all_offsets, size=chosen_diags, replace=False)
    data = rng.uniform(size=(chosen_diags, matrix_size))

    matrix = sparse.dia_array((data, offsets), shape=(matrix_size, matrix_size), dtype=np.float64)

    print("Symmetrizing matrix")
    matrix = -matrix - matrix.transpose()


    print("Making diagonally dominant")
    diag_sum = np.abs(np.copy(matrix.sum(axis=1)).flatten() - matrix.diagonal())
    diag_sum += 0.0001*diag_sum

    matrix.setdiag(diag_sum)

    rhs = rng.uniform(size=(matrix_size,1))

    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    nnz = matrix.nnz

    print("Solving")
    solution = np.linalg.solve(matrix.toarray(),rhs)

    print("Saving")
    with open(save_path + "K" + "_data_" + str(matsize) + ".bin", "wb") as f:
        f.write(matrix.data.reshape((1, -1)).tobytes())
    with open(save_path + "K" + "_col_indices_" + str(matsize) + ".bin", "wb") as f:
        indices = matrix.indices.astype(np.int32)
        f.write(indices.reshape((1, -1)).tobytes())
    with open(save_path + "K" + "_row_ptr_" + str(matsize) + ".bin", "wb") as f:
        indptr = matrix.indptr.astype(np.int32)
        f.write(indptr.reshape((1, -1)).tobytes())
    with open(save_path + "K" + "_rhs_" + str(matsize) + ".bin", "wb") as f:
        f.write(rhs.reshape((1, -1)).tobytes())
    with open(save_path + "K_solution_" + str(matsize) + ".bin", "wb") as f:
        f.write(solution.reshape((1, -1)).tobytes())

    # save matrix size and nnz
    with open(save_path + "K" + "_matrix_size_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(matrix_size, dtype=np.int32).tobytes())
    with open(save_path + "K" + "_nnz_" + str(matsize) + ".bin", "wb") as f:
        f.write(np.array(nnz, dtype=np.int32).tobytes())
