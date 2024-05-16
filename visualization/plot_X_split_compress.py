import numpy as np
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
    linewidth = 3
    elinewidth = 4
    capsize = 30
    captick = 5

    save_path ="/scratch/project_465000929/maederal/ACM_Poster/results/"

    names = ["X_not_split", "X_uncompressed", "X_compressed"]

    times = []
    for i, name in enumerate(names):
        times.append((np.loadtxt(save_path + name + "_" + str(1) + "_"+ str(0) +".txt").flatten()).tolist())
    times = np.array(times)
    print(times.shape)

    stds = np.std(times, axis=1)
    medians = np.median(times, axis=1)
    print(medians)
    confidence = 0.95
    interval = np.empty((2, times.shape[0]))
    for i in range(times.shape[0]):
        interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                        loc=np.median(times[i]),
                                        scale=st.sem(times[i]))
        

    yer = [interval[:, i].reshape((2, -1)) for i in range(times.shape[0])]
    for i in range(times.shape[0]):
        yer[i][0] = -yer[i][0] + medians[i]
        yer[i][1] = yer[i][1] - medians[i]

    barWidth = 0.8
    position0 = np.arange(int(times.shape[0]))
    position1 = position0 + barWidth

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 12)

    colors = ["darkblue", "dodgerblue",
                "darkgreen", "lime", "purple"]

    labels = [
        "Normal",
        "Split",  
        "+ Compressed"
    ]
    for i in range(len(names)):
        ax.bar(position0[i], medians[i], width=barWidth, color=colors[i],
                edgecolor="black", yerr=yer[i], capsize=capsize, label=labels[i], linewidth=linewidth)
        ax.errorbar(position0[i], medians[i], yerr=yer[i], fmt='o', color="black", capsize=capsize, elinewidth=elinewidth, capthick=captick, linewidth=linewidth)

    # general layout
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Decomposition")

    ax.legend()
    # ax.set_yscale('log')
    # ax.set_yticks([1e-1, 1, 1e1, 1e2, 1e3], minor=False)
    # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    ax.set_ylim(bottom=0)
    plt.savefig("X_split_compressed.png", bbox_inches='tight', dpi=300)
    plt.close('all')