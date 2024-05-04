import numpy as np
import scipy.sparse as sp

if __name__ == "__main__":
    data_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices"

    matsize = 100

    data = np.fromfile(data_path + "/K_data_" + str(matsize) + ".bin", dtype=np.float64)
    row_ptr = np.fromfile(data_path + "/K_row_ptr_" + str(matsize) + ".bin", dtype=np.int32)
    col_indices = np.fromfile(data_path + "/K_col_indices_" + str(matsize) + ".bin", dtype=np.int32)

    rhs = np.fromfile(data_path + "/K_rhs_" + str(matsize) + ".bin", dtype=np.float64)
    solution = np.fromfile(data_path + "/K_solution_" + str(matsize) + ".bin", dtype=np.float64)

    matrix_size = np.size(row_ptr) - 1
    nnz = np.size(data)
    print("Matrix size: ", matrix_size)
    print("Number of non-zero elements: ", nnz)

    assert np.size(row_ptr) == matrix_size + 1
    assert np.size(col_indices) == nnz
    assert np.size(rhs) == matrix_size
    assert np.size(solution) == matrix_size

    csr_matrix = sp.csr_matrix((data, col_indices, row_ptr), shape=(matrix_size, matrix_size))

    print("Solving system...")
    test_solution = sp.linalg.spsolve(csr_matrix, rhs)

    assert np.allclose(test_solution, solution)
