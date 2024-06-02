import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  
from matplotlib.lines import Line2D

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 50})
    matplotlib.rcParams['axes.linewidth'] =3
    matplotlib.rcParams['xtick.major.size'] = 10
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 10
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 5
    matplotlib.rcParams['ytick.minor.width'] = 1

    linewidth = 5
    elinewidth = 6
    capsize = 12
    captick = 5

    number_of_measured_steps = 1
    base_path = "/scratch/project_465000929/maederal/ACM_Poster/results/"
    images_path = ""

    legend_elements = \
    [
        Line2D([0], [0], color='red',lw=linewidth, label='RW: PPTP+Comp.', linestyle="dashed"),
        Line2D([0], [0], color='green', lw=linewidth, label='RW: PPTP', linestyle="dashed"),
        Line2D([0], [0], color='blue', lw=linewidth, label='RW: PTP', linestyle="dashed"),
        Line2D([0], [0], color='orange', lw=linewidth, label='RW: Allgather', linestyle="dashed")
    ]


    legend_elements2 = \
    [
        Line2D([0], [0], color='darkred',lw=linewidth, label='BW: PPTP+Comp.', linestyle="dotted"),
        Line2D([0], [0], color='darkgreen', lw=linewidth, label='BW: PPTP', linestyle="dotted"),
        Line2D([0], [0], color='darkblue', lw=linewidth, label='BW: PTP', linestyle="dotted")
    ]



    
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 12)

    leg = ax.legend(handles=legend_elements, loc="upper left")
    ax.add_artist(leg)
    ax.legend(handles=legend_elements2, loc="lower left")

    plt.savefig(images_path + "K_legend.png", bbox_inches='tight', dpi=300)
