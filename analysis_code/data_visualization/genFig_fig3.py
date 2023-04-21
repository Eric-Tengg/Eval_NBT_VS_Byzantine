import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from clink_simulator import *


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def genFig3():
    """
    对不同链路类型 DR, FPR, F1 下降的概率统计及CDF图
    """
    fig, axes_n = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
    linkKinds = ['root', 'internal', 'edge']
    for scn, axes in zip(['even', 'uneven'], axes_n):
        data_path = os.path.join(get_root_path(), 'analysis', f'statistical_analysis', scn)
        if scn == 'even':
            sections = [(i, round(i + .1, 1)) for i in np.linspace(.0, .9, 10).round(1)]
        else:
            sections = [(0, round(i + .1, 1)) for i in np.linspace(0.0, .9, 10).round(1)]
        path_set = np.linspace(1, 10, 10).astype(int)
        routing_matrix = pd.read_csv(os.path.join(get_root_path(), 'dataset', 'topo_breadth', 'topoSet4', '10-16.csv'),
                                     header=None).to_numpy()
        alg_set = ['scfs', 'clink', 'map']
        perf_set = ['dr', 'fpr', 'f1']

        plt.rcParams['axes.linewidth'] = 2
        totalData = {"DR": {"root": [], "internal": [], "edge": []},
                     "FPR": {"root": [], "internal": [], "edge": []},
                     "F1": {"root": [], "internal": [], "edge": []}}
        for sec_idx, section in enumerate(sections):
            # Read data
            # alg, linkKind, times
            DR, FPR, F1 = np.zeros((3, 3, 3, 20), dtype=float) if scn == "true" else np.zeros((3, 3, 3, 20 * 10),
                                                                                              dtype=float)
            for ax_idx in tqdm(range(20), desc=f"{section}"):  # draw plots based on section
                for alg_idx, alg_kind in enumerate(alg_set):  # one plot of different lines
                    if scn == 'true':
                        filePath = os.path.join(data_path, f'{section}-{ax_idx + 1}', 'true',
                                                f'linkstatus_{alg_kind}_high_eval.csv')
                        filedata = pd.read_csv(filePath).to_numpy().T

                        link_set = {'root': [1], 'intermediate': [2, 4, 7, 11, 13],
                                    'edge': [3, 5, 6, 8, 9, 10, 12, 14, 15, 16]}

                        data = np.zeros((3, 3), dtype=float)  # DR,
                        for idx, link_kind in enumerate(link_set):
                            linkSet = (np.array(link_set[link_kind]) - 1).tolist()
                            data[:, idx] = np.mean(filedata[:, linkSet], axis=1)

                        for idx, perf in enumerate(perf_set):  # the data Set of DR, FPR, F1
                            locals()[perf.upper()][alg_idx, :, ax_idx] = data[idx]
                    elif scn == 'attack':
                        for path in path_set:  # no-difference in 'path'
                            filePath = os.path.join(data_path, f'{section}-{ax_idx + 1}', f'path-{path}',
                                                    f'linkstatus_attacked_{alg_kind}_high_eval.csv')
                            filedata = pd.read_csv(filePath).to_numpy().T

                            link_set = {'root': [1], 'intermediate': [2, 4, 7, 11, 13],
                                        'edge': [3, 5, 6, 8, 9, 10, 12, 14, 15, 16]}

                            data = np.zeros((3, 3), dtype=float)
                            for idx, link_kind in enumerate(link_set):
                                linkSet = (np.array(link_set[link_kind]) - 1).tolist()
                                data[:, idx] = np.mean(filedata[:, linkSet], axis=1)

                            DR[alg_idx, :, ax_idx * 10 + (path - 1)] = data[alg_idx]
                            FPR[alg_idx, :, ax_idx * 10 + (path - 1)] = data[alg_idx]
                            F1[alg_idx, :, ax_idx * 10 + (path - 1)] = data[alg_idx]
                    else:
                        for path in path_set:  # no-difference in 'path'
                            filePath = os.path.join(data_path, f'{section}-{ax_idx + 1}', f'path-{path}',
                                                    f'{alg_kind}_based_on_linkKinds_relative.csv')
                            df = pd.read_csv(filePath, index_col=0)

                            DR[alg_idx, :, ax_idx * 10 + (path - 1)] = \
                                df.loc[df.index == 'DR'].to_numpy()
                            FPR[alg_idx, :, ax_idx * 10 + (path - 1)] = \
                                df.loc[df.index == 'FPR'].to_numpy()
                            F1[alg_idx, :, ax_idx * 10 + (path - 1)] = \
                                df.loc[df.index == 'F1'].to_numpy()

            for kind_idx in range(3):
                totalData["DR"][linkKinds[kind_idx]] += list(np.hstack([DR[m][kind_idx] for m in range(3)]))
                totalData["FPR"][linkKinds[kind_idx]] += list(np.hstack([FPR[m][kind_idx] for m in range(3)]))
                totalData["F1"][linkKinds[kind_idx]] += list(np.hstack([F1[m][kind_idx] for m in range(3)]))

        # Create subplots
        bin_edges = np.linspace(-1.0, 1.0, 101)  # adjust the range and number of bins as needed
        for ax_idx, ax in enumerate(axes):  # one plot of different performance index —— in cols
            # plot the cumulative histogram
            epsilon = 1e-8
            for n in range(3):  # Different kinds of links
                perf_key = perf_set[ax_idx].upper()
                link_key = linkKinds[n]
                data = totalData[perf_key][link_key]
                ax.hist(data, bins=bin_edges, density=True, histtype='step', cumulative=1, linewidth=2)
                # Calculate the mean and standard deviation of the data
                mu, sigma = np.mean(data), np.std(data)

                # Create a normal distribution with the calculated parameters
                norm_dist = stats.norm(loc=mu, scale=sigma + epsilon)

                # Calculate the CDF of the normal distribution
                cdf = norm_dist.cdf(bin_edges)

                # Plot the CCDF of the expected distribution
                ax.plot(bin_edges, cdf, '--', linewidth=1)
                ax.tick_params(axis='x', labelsize=13)
                ax.tick_params(axis='y', labelsize=13)
            ax.grid(True)
            if scn == 'uneven':
                ax.set_xlabel(f'Performance Degradation of {perf_set[ax_idx].upper()}', fontsize=15, style='italic')
            if ax_idx == 1:
                ax.set_title(scn, fontsize=15, style='italic')
            ax.set_ylabel('CDF', fontsize=15, style='italic')
            ax.set_xlim(-1.02, 1.02)
            ax.axhline(y=0.5, color='red', linewidth=2, linestyle='--', alpha=0.5)

    # Adjust the layout
    fig.legend([f"{typeLine} —— {linkKind}" for linkKind in linkKinds for typeLine in ["Discrete CDF", "Expected CDF"]],
               loc='upper center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1.11))
    plt.tight_layout()
    # Specify the path where you want to save the plot
    output_path = os.path.join(get_root_path(), 'analysis', 'figures', "fig3.png")
    # Save the plot to the specified path
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    # Close the plot to release resources
    plt.show()

if __name__ == '__main__':
    genFig3()
