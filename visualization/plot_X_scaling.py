import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

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
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green"
    ]
    labels = [
        "Overlap",
        "Single Kernel",
        "Split"
    ]

    paths = [
        base_path,
        base_path,
        base_path
    ]



    reference_path = base_path + "X_singlekernel_compressed1_1_0.txt"
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

    ax.set_ylabel("Speedup")
    ax.set_xlabel("GCDs")
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend()
    # ax.set_ylim(bottom=0,top=3.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, minor=False)
    ax.set_xticklabels(sizes, minor=False)
    plt.savefig(images_path + "X_scaling.png", bbox_inches='tight', dpi=300)
