import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  
from matplotlib.ticker import FormatStrFormatter
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

    base_path = "/scratch/project_465000929/maederal/ACM_Poster/results/"
    images_path = ""
    sizes = [1, 2, 4, 8, 16, 32]

    method_names = [
        "X_not_split_overlap",
        "X_not_split_singelkernel",
        "X_singlekernel_compressed1"
    ]

    method_names = [
        "X_not_split_singelkernel",
        "X_singlekernel_compressed3"
    ]
    colors = [
        "tab:blue",
        "darkgreen"
    ]
    labels = [
        "Non Split",
        "Split"
    ]

    paths = [
        base_path,
        base_path
    ]
   
    num = "2"

    # method_names = [
    #     "X_not_split_singelkernel"
    # ]
    # colors = [
    #     "tab:blue"
    # ]
    # labels = [
    #     "Non Split"
    # ]

    # paths = [
    #     base_path
    # ]

    # num = "1"

    reference_path = base_path + "X_not_split_singelkernel_1_0.txt"
    reference_time = np.loadtxt(reference_path).flatten()
    median_reference_time = np.median(reference_time)
    median_reference_time = np.min([median_reference_time])

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 12)
    for i, method_name in enumerate(method_names):

        times = [[] for j in range(len(sizes))]
        for j in range(len(sizes)):
            times[j] += (np.loadtxt(paths[i] + method_name + "_" + str(sizes[j]) + "_"+ str(0) +".txt").flatten()).tolist()
        
        times = [median_reference_time/np.array(times[j]) for j in range(len(sizes))]

    
        stds = []
        medians = []
        interval = []
        confidence = 0.95
        for j in range(len(sizes)):
            stds.append(np.std(times[j]))
            medians.append(np.median(times[j]))
            interval.append(st.t.interval(confidence=confidence, df=len(times[j])-1,
                    loc=np.median(times[j]),
                    scale=st.sem(times[j])))
            

        yer = []
        for j in range(len(sizes)):
            yer.append(np.copy(interval[j]))
            yer[j][0] = -yer[j][0] + medians[j]
            yer[j][1] = yer[j][1] - medians[j]

        yer = np.array(yer).T
        x = np.array(sizes)
        ax.plot(x, medians, label=labels[i], color=colors[i], linestyle='dashed', linewidth=linewidth)
        plt.errorbar(x, medians, yerr=np.squeeze(yer), color=colors[i], capsize=capsize, barsabove=True, marker='x', linestyle='None', linewidth=linewidth, elinewidth=elinewidth, capthick=captick)

        print(medians)

    ax.set_ylabel("Speedup")

    # legend_elements2 = \
    # [
    #     Line2D([0], [0], color='black', lw=linewidth, label='Rowwise', linestyle="dashed"),
    # ]

    # leg = ax.legend(handles=legend_elements2, loc="lower right")
    # ax.add_artist(leg)
    ax.legend()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylim(bottom=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, minor=False)
    ax.set_xticklabels(sizes, minor=False)


    ax.set_yticks([1,3,5,7,9], minor=False)
    ax.set_yticklabels([1,3,5,7,9], minor=False)

    secax = ax.secondary_xaxis("bottom")
    secax.set_xscale("log", base=2)
    # secax.xaxis.set_label_position('bottom')
    # secax.xaxis.set_ticks_position('bottom')
    # secax.spines['bottom'].set_position(('outward', 20))

    # Set secondary x-axis ticks
    secax.spines['bottom'].set_position(('outward', 45))
    secax.set_xticks(sizes)
    # secax.set_xticklabels(sizes)
    secax.set_xticklabels(['(1)', '(1)', '(1)', '(1)', '(2)', '(4)'])

    # Hide the secondary axis line
    secax.spines['top'].set_visible(False)
    secax.spines['bottom'].set_visible(False)
    secax.spines['right'].set_visible(False)
    secax.spines['left'].set_visible(False)
    secax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

    secax.set_xlabel("GCDs (Nodes)")


    plt.savefig(images_path + "X_scaling"+num+".png", bbox_inches='tight', dpi=300)
