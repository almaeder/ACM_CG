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

    matsize = 1600
    row_ptr = np.fromfile(data_path + "K_row_ptr_" + str(matsize) + ".bin", dtype=np.int32)
    col_indices = np.fromfile(data_path + "K_col_indices_" + str(matsize) + ".bin", dtype=np.int32)

    sparse_matrix = sparse.csr_matrix(
        (np.ones_like(col_indices, dtype=float), col_indices, row_ptr), dtype=float)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    plt.spy(sparse_matrix, markersize=0.0001)

    ax.set_yticks([], minor=False)
    ax.set_xticks([], minor=False)
    ax.get_yaxis().get_offset_text().set_visible(False)
    # ax.get_xaxis().get_offset_text().set_position((0.5,0))
    # plt.axis('off')
    plt.savefig("K_matrix.png", bbox_inches='tight', dpi=300)

    plt.close('all')
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    plt.spy(sparse_matrix[:50000,:50000], markersize=0.0001, aspect='equal')

    ax.set_yticks([], minor=False)
    ax.set_xticks([], minor=False)
    ax.get_yaxis().get_offset_text().set_visible(False)
    ax.get_xaxis().get_offset_text().set_position((0.5,0))
    # plt.axis('off')
    plt.savefig("K_matrix_zoom.png", bbox_inches='tight', dpi=1000)