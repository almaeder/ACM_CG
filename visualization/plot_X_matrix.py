import numpy as np
import matplotlib.pyplot as plt
import matplotlib  
from scipy import sparse

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 40})
    matplotlib.rcParams['axes.linewidth'] =3
    matplotlib.rcParams['xtick.major.size'] = 10
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 10
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 5
    matplotlib.rcParams['ytick.minor.width'] = 1
    linewidth = 3
    elinewidth = 4
    capsize = 30
    captick = 5

    data_path = "/scratch/project_465000929/maederal/ACM_Poster/matrices/"

    matsize = 401
    row_ptr = np.fromfile(data_path + "X_row_ptr_" + str(matsize) + ".bin", dtype=np.int32)
    col_indices = np.fromfile(data_path + "X_col_indices_" + str(matsize) + ".bin", dtype=np.int32)

    row_ptr_subblock = np.fromfile(data_path + "X_ptr_uncompressed.bin", dtype=np.int32)
    col_indices_subblock = np.fromfile(data_path + "X_indices_uncompressed.bin", dtype=np.int32)

    matrix_size = row_ptr.size -1
    assert row_ptr.size == row_ptr_subblock.size

    sparse_matrix = sparse.csr_matrix(
        (np.ones_like(col_indices, dtype=float), col_indices, row_ptr), shape=(matrix_size,matrix_size), dtype=float)

    sparse_matrix_subblock = sparse.csr_matrix(
        (np.ones_like(col_indices_subblock, dtype=float), col_indices_subblock, row_ptr_subblock), shape=(matrix_size,matrix_size), dtype=float)

    print(sparse_matrix_subblock.nnz / (matrix_size**2) )

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    
    mat = sparse_matrix_subblock
    # plt.spy(mat, markersize=0.0000001, color="forestgreen", aspect='equal', alpha=0.9)
    fig.patch.set_alpha(0.0)
    plt.spy(sparse_matrix, markersize=0.0001, aspect='equal')

    ax.set_yticks([], minor=False)
    ax.set_xticks([], minor=False)
    ax.get_yaxis().get_offset_text().set_visible(False)
    ax.get_xaxis().get_offset_text().set_position((0.5,0))
    
    # plt.axis('off')
    plt.savefig("X_matrix3.png", bbox_inches='tight', dpi=1000, transparent=True)

    # plt.close('all')
    # fig, ax = plt.subplots()
    # fig.set_size_inches(12, 12)
    # plt.spy(mat[:50000,:50000], markersize=0.0000001, color="lightgreen", aspect='equal', alpha=0.5)
    # plt.spy(sparse_matrix[:50000,:50000], markersize=0.0001, aspect='equal')

    # ax.set_yticks([], minor=False)
    # ax.set_xticks([], minor=False)
    # ax.get_yaxis().get_offset_text().set_visible(False)
    # ax.get_xaxis().get_offset_text().set_position((0.5,0))
    # # plt.axis('off')
    # plt.savefig("X_matrix_zoom.png", bbox_inches='tight', dpi=1000)

