import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices/"

    matsize = 100

    row_ptr = np.fromfile(data_path + "X_subblock_row_ptr_" + str(matsize) + ".bin", dtype=np.int32)
    col_indices = np.fromfile(data_path + "X_subblock_col_indices_" + str(matsize) + ".bin", dtype=np.int32)

    sparse_matrix = sparse.csr_matrix(
        (np.ones_like(col_indices, dtype=float), col_indices, row_ptr), dtype=float)

    plt.spy(sparse_matrix.toarray())
    plt.savefig("subblock_spy.png")
    