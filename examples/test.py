import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
import statistics
from scipy import stats
import os
import numpy as np
from tqdm import tqdm


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def draw_fig2():
    # Set seaborn style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    data = {}
    folderPath = os.path.join(data_path, 'fig2')
    Scenario = ["even", "uneven"]
    
    for ax, (index, scn) in zip(axes, enumerate(['even', 'uneven'])):
        with open(os.path.join(folderPath, f'{scn}.json'), 'r') as infile:
            filedata = json.load(infile)
        if scn == 'even':
            intervals = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        else:
            intervals = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        alg_set = ['SCFS', 'CLINK', 'MAP']

        true_dr = {}
        for interval in intervals:
            true_dr[str(interval)] = np.mean(
                [statistics.mean(filedata['true']['DR'][alg][str(interval)].values()) for alg in alg_set])
        true_fpr = {}
        for interval in intervals:
            true_fpr[str(interval)] = np.mean(
                [statistics.mean(filedata['true']['FPR'][alg][str(interval)].values()) for alg in alg_set])
        true_f1 = {}
        for interval in intervals:
            true_f1[str(interval)] = np.mean(
                [statistics.mean(filedata['true']['F1'][alg][str(interval)].values()) for alg in alg_set])

        path_set = list(range(1, 11))
        att_dr = {}
        for interval in intervals:
            att_dr[str(interval)] = np.mean(
                [statistics.mean(filedata['attack']['DR'][alg][str(interval)][f"sample{i + 1}"][f"path{path}"].values())
                 for i in range(20) for alg in alg_set for path in path_set])
        att_fpr = {}
        for interval in intervals:
            att_fpr[str(interval)] = np.mean(
                [statistics.mean(
                    filedata['attack']['FPR'][alg][str(interval)][f"sample{i + 1}"][f"path{path}"].values())
                    for i in range(20) for alg in alg_set for path in path_set])
        att_f1 = {}
        for interval in intervals:
            att_f1[str(interval)] = np.mean(
                [statistics.mean(filedata['attack']['F1'][alg][str(interval)][f"sample{i + 1}"][f"path{path}"].values())
                 for i in range(20) for alg in alg_set for path in path_set])

        data[scn] = {
            'true': [statistics.mean(true_dr.values()), statistics.mean(true_fpr.values()),
                     statistics.mean(true_f1.values())],
            'attack': [statistics.mean(att_dr.values()), statistics.mean(att_fpr.values()),
                       statistics.mean(att_f1.values())],
        }

        performance_indices = ['DR', 'FPR', 'F1']
        colors = ['#1f77b4', '#ff7f0e']
        bar_width = 0.35

        hatch_patterns = ['/', 'x', '+']

        for idx, (key, values) in enumerate(data[scn].items()):
            x_positions = [i + idx * bar_width for i in range(len(performance_indices))]
            bars = ax.bar(x_positions, values, width=bar_width, color=colors[idx], label=key,
                          edgecolor='black', linewidth=1, hatch=hatch_patterns[idx], alpha=0.8)

            # Add data labels on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=12)

        #     ax.set_ylabel('Values', fontsize=15, style='italic')
        ax.set_ylim(0, 0.8)
        ax.set_title(Scenario[index], fontsize=15, style='italic')

        ax.set_xticks([i + bar_width / 2 for i in range(len(performance_indices))])
        ax.set_xticklabels(performance_indices, fontsize=15, style='italic')

        ax.tick_params(axis='y', labelsize=15, width=10)

        # Add grid lines
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.rcParams['axes.linewidth'] = 2
    plt.tight_layout()
    # Add a shared legend below the plot and set it horizontally
    fig.legend(['With no attack', 'Byzantine attack'], loc='upper center', ncol=2, fontsize=15,
               bbox_to_anchor=(0.5, 1.21))

    # Adjust the space for the shared legend
    plt.subplots_adjust(top=1)

    plt.savefig(os.path.join(figure_savePath, f"fig2.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig3():
    """
    对不同链路类型 DR, FPR, F1 下降的概率统计及CDF图
    """
    scn = 'differ'
    folder_path = os.path.join(data_path, 'fig3')
    
    sections = [(i, round(i + 0.1, 1)) for i in np.linspace(0.0, 0.4, 3).round(1)]
    path_set = np.linspace(1, 10, 10).astype(int)
    routing_matrix = pd.read_csv(os.path.join(get_root_path(), 'datasets', 'tree_topo', '10-16.csv'), header=None) \
        .to_numpy()
    linkKinds = ['root', 'internal', 'edge']
    alg_set = ['scfs', 'clink', 'map']
    perf_set = ['dr', 'fpr', 'f1']

    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    plt.rcParams['axes.linewidth'] = 2
    totalData = {"DR": {"root": [], "internal": [], "edge": []},
                 "FPR": {"root": [], "internal": [], "edge": []},
                 "F1": {"root": [], "internal": [], "edge": []}}
    for sec_idx, section in enumerate(sections):
        # Read data
        # alg, linkKind, times
        DR, FPR, F1 = np.zeros((3, 3, 3, 20), dtype=float) if scn == "true" else np.zeros((3, 3, 3, 20 * 10),
                                                                                          dtype=float)
        for ax_idx in range(20):  # draw plots based on section
            for alg_idx, alg_kind in enumerate(alg_set):  # one plot of different lines
                if scn == 'true':
                    filePath = os.path.join(folder_path, f'{section}-{ax_idx + 1}', 'true',
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
                        filePath = os.path.join(folder_path, f'{section}-{ax_idx + 1}', f'path-{path}',
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
                        filePath = os.path.join(folder_path, f'{section}-{ax_idx + 1}', f'path-{path}',
                                                f'{alg_kind}_based_on_linkKinds_relative.csv')
                        df = pd.read_csv(filePath, index_col=0)
                        DR[alg_idx, :, ax_idx * 10 + (path - 1)] = df.loc[df.index == 'DR'].to_numpy()
                        FPR[alg_idx, :, ax_idx * 10 + (path - 1)] = df.loc[df.index == 'FPR'].to_numpy()
                        F1[alg_idx, :, ax_idx * 10 + (path - 1)] = df.loc[df.index == 'F1'].to_numpy()

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
        ax.set_xlabel(f'Performance Degradation of {perf_set[ax_idx].upper()}', fontsize=15, style='italic')
        ax.set_ylabel('CDF', fontsize=15, style='italic')
        ax.set_xlim(-0.5, 1.02)

    # Adjust the layout
    fig.legend(
        [f"{typeLine} —— {linkKind}" for linkKind in linkKinds for typeLine in ["Discrete CDF", "Expected CDF"]],
        loc='upper center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1.2))
    plt.tight_layout()
    # Save the plot to the specified path
    plt.savefig(os.path.join(figure_savePath, 'fig3.png'), format='png', dpi=300, bbox_inches='tight')
    # Close the plot to release resources
    plt.close()


def draw_fig4_a():
    Scenario = ["even", "uneven"]
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
    for ax, (index, scn) in zip(axes, enumerate(['even', 'uneven'])):
        folderPath = os.path.join(data_path, 'fig4')
        with open(os.path.join(folderPath, f'{scn}.json'), 'r') as infile:
            filedata = json.load(infile)
        if scn == 'even':
            intervals = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        else:
            intervals = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        alg_set = ['SCFS', 'CLINK', 'MAP']
        path_set = list(range(1, 11))

        plotData = {'true': [], 'attack': []}
        for alg in alg_set:
            plotData['true'].append(
                statistics.mean([statistics.mean(filedata['true']['F1'][alg][str(interval)].values())
                                 for interval in intervals]))
            plotData['attack'].append(statistics.mean(
                [statistics.mean(filedata['attack']['F1'][alg][str(interval)][f"sample{i + 1}"][f"path{path}"].values())
                 for interval in intervals for i in range(20) for path in path_set]))

        performance_indices = ['SCFS', 'CLINK', 'MAP']
        colors = ['#1f77b4', '#ff7f0e']
        bar_width = 0.35

        hatch_patterns = ['/', 'x', '+']

        for idx, (key, values) in enumerate(plotData.items()):
            x_positions = [i + idx * bar_width for i in range(len(performance_indices))]
            bars = ax.bar(x_positions, values, width=bar_width, color=colors[idx], label=key,
                          edgecolor='black', linewidth=1, hatch=hatch_patterns[idx], alpha=0.8)

            # Add data labels on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        #     ax.set_ylabel('Identical Performance', fontsize=15, style='italic')
        ax.set_ylim(0, 0.88)
        ax.set_title(Scenario[index], fontsize=15, style='italic')
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks([i + bar_width / 2 for i in range(len(performance_indices))])
        ax.set_xticklabels(performance_indices, fontsize=15, style='italic')

        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.text(-0.02, 0.6, 'Performance', va='center', rotation='vertical', fontsize=15, style='italic')
    # Add a shared legend below the plot and set it horizontally
    fig.legend(['With no attack', 'Byzantine attack'], loc='lower center', ncol=2, fontsize=15,
               bbox_to_anchor=(0.5, 0.98))
    plt.rcParams['axes.linewidth'] = 2
    # Adjust the space for the shared legend
    plt.subplots_adjust(bottom=0.23)

    plt.savefig(os.path.join(figure_savePath, f"fig4_a.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig4_b():
    # 算法间的不同
    alg_set = ['SCFS', 'CLINK', 'MAP']
    data = {}
    folderPath = os.path.join(data_path, 'fig4')
    for scn in ['even', 'uneven']:
        data[scn] = []
        with open(os.path.join(folderPath, f'{scn}.json'), 'r') as infile:
            filedata = json.load(infile)
        if scn == 'even':
            intervals = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        else:
            intervals = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]

        true_f1 = {}
        for alg in alg_set:
            true_f1[alg] = np.mean(
                [statistics.mean(filedata['true']['F1'][alg][str(interval)].values()) for interval in intervals])

        path_set = list(range(1, 11))

        att_f1 = {}
        for alg in alg_set:
            att_f1[alg] = np.mean(
                [statistics.mean(filedata['attack']['F1'][alg][str(interval)][f"sample{i + 1}"][f"path{path}"].values())
                 for interval in intervals for i in range(20) for path in path_set])

        for alg in alg_set:
            data[scn].append((true_f1[alg] - att_f1[alg]) / true_f1[alg])

    with open(os.path.join(folderPath, f'part1_2.1.json'), 'w') as outfile:
        json.dump(data, outfile)
    plotdata = {}
    for idx, alg in enumerate(alg_set):
        plotdata[alg] = [data['even'][idx], data['uneven'][idx]]

    fig, ax = plt.subplots(figsize=(8, 3))
    performance_indices = ['even', 'uneven']
    colors = ['#1f77b4', '#ff7f0e', '#426f42']
    bar_height = 0.3

    hatch_patterns = ['/', 'x', '+']

    for idx, (key, values) in enumerate(plotdata.items()):
        y_positions = [i + idx * bar_height for i in range(len(performance_indices))]
        bars = ax.barh(y_positions, values, height=bar_height, color=colors[idx], label=key,
                       edgecolor='black', linewidth=1, hatch=hatch_patterns[idx], alpha=0.8)

        # Add data labels on top of the bars
        for bar in bars:
            width = bar.get_width()
            label = int(width * 100)
            ax.annotate(f'{label}%',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=12)

    # ax.set_xlabel('Relative Performance Degradation', fontsize=15, style='italic')
    ax.set_xlim(0, 0.3)
    # ax.set_title('Relative Degradation of Attack Different Algorithms', fontsize=15, style='italic')

    ax.set_yticks([i + bar_height / 2 + 0.35 for i in range(len(performance_indices))])
    ax.set_yticklabels(["even", "uneven"], fontsize=15, style='italic', rotation='vertical')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.rcParams['axes.linewidth'] = 2
    # Add grid lines
    # ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(0, 0.31, 0.05))
    ax.set_xticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '30%'], fontsize=13, style='italic')
    fig.legend(['SCFS', 'CLINK', 'MAP'], loc='lower center', ncol=3, fontsize=13, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(figure_savePath, f"fig4_b.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig5():
    fig, axes = plt.subplots(nrows=2, figsize=(6, 6), sharey=True)
    plt.rcParams['axes.linewidth'] = 2
    folder_path = os.path.join(data_path, 'fig5')
    
    for ax, (index, scn) in zip(axes, enumerate(['even', 'uneven'])):
        if scn == 'even':
            intervals = [(i, (i + 0.1).round(1)) for i in np.linspace(0.0, 0.9, 9).round(1)]
        else:
            intervals = [(0, (i + 0.1).round(1)) for i in np.linspace(0.0, 0.9, 9).round(1)]
        # 无关变量设置
        freq = 1.0
        path = 3
        with open(os.path.join(folder_path, f'{scn}_freq{freq}_path{path}.json'), 'r') as outfile:
            eff = json.load(outfile)
        # 画图
        sns.set_style("whitegrid")
        algorithm_labels = ["SCFS", "CLINK", "MAP"]
        n_attacks = len(eff['scfs'])
        x = np.arange(n_attacks)
        lineStyle = ['-.', '-', '--']
        palette = ['#FE8228', '#279F27', '#D62726']

        for algorithm_label, (idx, algorithm_eff) in zip(algorithm_labels, enumerate(eff.values())):
            ax.plot(x, algorithm_eff, label=algorithm_label, linestyle=lineStyle[idx], linewidth=2, color=palette[idx])
            ax.set_xticks(x)

        x_ticks = [str(interval) for interval in intervals]
        ax.set_xticklabels(x_ticks, fontsize=12, rotation=20)
        ax.set_ylim(-0.5, 1)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='y', labelsize=12)

    # Set the x-label and y-label for the figure
    fig.text(0.53, -0.02, 'The uniform distribution for Link prior congestion probability', ha='center', style='italic',
             fontsize=13)
    fig.text(-0.02, 0.29, 'Performance Degradation', va='center', rotation='vertical', style='italic', fontsize=13)
    fig.text(-0.02, 0.79, 'Performance Degradation', va='center', rotation='vertical', style='italic', fontsize=13)
    plt.tight_layout()

    fig.legend(["SCFS", "CLINK", "MAP"], loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 0.98))

    plt.savefig(os.path.join(figure_savePath, f'fig5.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_fig6():
    folder_path = os.path.join(data_path, 'fig6')
    values = np.zeros((3, 10))

    with open(os.path.join(folder_path, 'lowCongestionLevel.json'), 'r') as f:
        a = json.load(f)
    values[0] = np.array([np.mean([a[key][i] for key in a.keys()]) for i in range(10)])

    with open(os.path.join(folder_path, 'middleCongestionLevel.json'), 'r') as f:
        b = json.load(f)
    values[1] = np.array([np.mean([b[key][i] for key in b.keys()]) for i in range(10)])

    with open(os.path.join(folder_path, 'highCongestionLevel.json'), 'r') as f:
        c = json.load(f)
    values[2] = np.array([np.mean([c[key][i] for key in c.keys()]) for i in range(10)])

    # 生成数据
    freqs = [f"{i}%" for i in np.arange(10, 101, 10)]
    intervals = ['Low', 'Middle', 'High']

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(values, cmap="twilight", fmt='.2f', xticklabels=freqs, yticklabels=intervals, ax=ax)
    ax.set_xlabel('Attack Frequency', fontsize=15, style='italic')
    ax.set_ylabel('Congestion Level', fontsize=15, style='italic')
    bar_width = 0.35
    ax.set_xticks([i + bar_width / 2 + 0.3 for i in range(len(freqs))])
    ax.set_xticklabels(freqs, fontsize=13, style='italic')

    ax.set_yticks([i + bar_width / 2 + 0.3 for i in range(len(intervals))])
    ax.set_yticklabels(intervals, fontsize=13, style='italic')

    plt.savefig(os.path.join(figure_savePath, f"fig6.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig7():
    # 探讨频率
    # 设置无关变量
    def draw_fig7_a():
        nonlocal ax
        freq = 1.0
        interval = (0.0, 0.1)
        # 区间数
        with open(os.path.join(folderPath, f"a_interval{interval}_freq{freq}.json"), 'r') as outfile:
            cnt = json.load(outfile)

        # 画出 CDF 图

        algorithm_labels = ["SCFS", "CLink", "MAP"]

        # 使用其他内置样式
        # plt.style.use('ggplot')

        # 定义线条样式和颜色
        linestyles = ['-', '--', '-.']
        palette = ['#D62726', '#FE8228', '#279F27']
        for i, (alg, data) in enumerate(zip(algorithm_labels, cnt.values())):
            # 对数据进行排序
            sorted_data = np.sort(data)
            # 计算累积概率
            cum_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            # 画出CDF图
            ax[0].plot(sorted_data, cum_prob, linestyle=linestyles[i], linewidth=2, color=palette[i], label=alg)
        ax[0].spines['bottom'].set_color('black')
        ax[0].spines['top'].set_color('black')
        ax[0].spines['left'].set_color('black')
        ax[0].spines['right'].set_color('black')
        ax[0].set_xlabel('Relative Depth', fontsize=15, style='italic')
        ax[0].set_ylabel('CDF', fontsize=15, style='italic')
        ax[0].tick_params(axis='both', labelsize=15)

    def draw_fig7_b():
        """
            data structure :
            {
            '(0.0, 0.1)': {
                '(0.0, 0.2)': [attack_effectivity1, attack_effectivity2, ...],
                '(0.2, 0.4)': [attack_effectivity1, attack_effectivity2, ...],
                '(0.4, 0.6)': [attack_effectivity1, attack_effectivity2, ...],
                '(0.6, 0.8)': [attack_effectivity1, attack_effectivity2, ...],
                '(0.8, 1.0)': [attack_effectivity1, attack_effectivity2, ...]
            }
        }
        """
        nonlocal ax
        interval = (0.0, 0.1)
        freq = 1.0

        # Load dictionary from a JSON file
        with open(os.path.join(folderPath, f"b_interval{interval}_freq{freq}.json"), 'r') as infile:
            section_results = json.load(infile)

        # Prepare the data for plotting
        data_for_plotting = []

        for alg, alg_data in section_results.items():
            for relative_breadth, attack_effects in alg_data.items():
                for effect in attack_effects:
                    data_for_plotting.append({
                        'Algorithm': alg,
                        'Relative Breadth': relative_breadth,
                        'Attack Effect': effect
                    })

        # Create a DataFrame for easier plotting with seaborn
        df = pd.DataFrame(data_for_plotting)

        # Create the line plot
        sns.set_style("whitegrid")
        palette = ['#D62726', '#FE8228', '#279F27']
        sns.lineplot(data=df, x='Relative Breadth', y='Attack Effect', hue='Algorithm', style='Algorithm',
                     errorbar=None, legend=False, markers=True, linewidth=2, palette=palette, ax=ax[1])
        ax[1].spines['bottom'].set_color('black')
        ax[1].spines['top'].set_color('black')
        ax[1].spines['left'].set_color('black')
        ax[1].spines['right'].set_color('black')
        ax[1].set_xlabel('Relative Breadth ', fontsize=15, style='italic')
        ax[1].set_ylabel('Performance Degaradtion', fontsize=15, style='italic')
        ax[1].tick_params(axis='both', labelsize=12)
        shown_intervals = ["0.00", "0.25", "0.50", "0.75", "1.00"]
        ax[1].set_xticks([i * 2 for i in range(len(shown_intervals))])
        ax[1].set_xticklabels(shown_intervals, fontsize=15)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    folderPath = os.path.join(data_path, 'fig7')
    draw_fig7_a()
    draw_fig7_b()
    fig.legend(["SCFS", "CLINK", "MAP"], loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 0.9))
    plt.subplots_adjust(wspace=0.25, hspace=None)
    plt.savefig(os.path.join(figure_savePath, f"fig7.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig8():
    """
    由先验概率得到Y概率，遍历所有攻击情况(from path-1 to path-10),得到理论攻击情况
    """
    folderPath = os.path.join(data_path, 'fig8')
    filePath = os.path.join(folderPath, f"interval(0.0, 0.1)_freq1.0.csv")
    df = pd.read_csv(filePath)
    df['Cnt'] = [f"{i}%" for _ in range(3) for i in np.linspace(0, 100, 11).astype(int)]
    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 4))

    alg_set = ['SCFS', 'CLINK', 'MAP']

    performance_indices = [f'{i}%' for i in range(0, 101, 10)]
    colors = ['#1f77b4', '#ff7f0e', '#426f42']
    bar_width = 0.2

    hatch_patterns = ['/', 'x', '+']

    for idx in range(3):
        values = df[df['Alg'] == alg_set[idx]]['Value']

        x_positions = [i + idx * bar_width for i in range(len(performance_indices))]

        ax.bar(x_positions, values, width=bar_width, color=colors[idx], label=performance_indices[idx],
               edgecolor='black', linewidth=1, hatch=hatch_patterns[idx], alpha=0.8)

    ax.set_ylabel('Performance Degradation', fontsize=15, style='italic')
    ax.set_ylim(0, 1.02)

    ax.set_xticks([i + bar_width / 2 for i in range(len(performance_indices))])
    ax.set_xticklabels(performance_indices, fontsize=15, style='italic')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    plt.rcParams['axes.linewidth'] = 2
    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    # Add a shared legend below the plot and set it horizontally
    fig.legend(alg_set, loc='lower center', ncol=3, fontsize=15, bbox_to_anchor=(0.5, 0.98))

    # Adjust the space for the shared legend
    plt.savefig(os.path.join(figure_savePath, f"fig8.png"), format="png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_fig9():
    folderPath = os.path.join(data_path, 'fig9')

    def interval_exp():
        """
        攻击频率为1，攻击路径为path3
        """
        intervals = [(i, (i + 0.1).round(1)) for i in np.linspace(0.0, 0.9, 9).round(1)]
        filePath = os.path.join(folderPath, "interval_exp.csv")
        att_eff = pd.read_csv(filePath).to_numpy().reshape(1, -1)[0]
        cor = np.corrcoef([interval[1] for interval in intervals], att_eff)[0][1]
        return cor

    def freq_exp(interval):
        """
        概率区间为(0.0, 0.1), 攻击路径为path3
        """
        freqSet = np.linspace(0.1, 1.0, 10).round(1)
        filePath = os.path.join(folderPath, f"freq_exp_{interval}.csv")
        att_eff = pd.read_csv(filePath).to_numpy().reshape(1, -1)[0]
        cor = np.corrcoef(freqSet, att_eff)[0][1]
        return cor

    def rd_exp(interval):
        """
        概率区间为(0.0, 0.1), 攻击频率为1.0
        """
        filePath = os.path.join(folderPath, f"rd_exp_{interval}.csv")
        sectionNum = 5  # 区间个数 —— 0.00 ~ 0.20， 0.20 ~ 0.40, 0.40 ~ 0.60, 0.60 ~ 0.80, 0.80 ~ 1.00
        cnt = pd.read_csv(filePath).to_numpy().reshape(1, -1)[0]
        cor = np.corrcoef(cnt, np.arange(1, sectionNum + 1, 1))[0][1]

        return cor

    def rb_exp(interval):
        filePath = os.path.join(folderPath, f"rb_exp_{interval}.csv")
        att_eff = pd.read_csv(filePath).to_numpy().reshape(1, -1)[0]

        cor = np.corrcoef(att_eff, np.linspace(0.2, 1.0, 5))[0][1]
        return cor

    def conDegree_exp(interval):
        """
        由先验概率得到Y概率，遍历所有攻击情况(from path-1 to path-10),得到理论攻击情况
        """
        filePath = os.path.join(folderPath, f"conDegree_exp_{interval}.csv")
        att_eff = pd.read_csv(filePath).to_numpy().reshape(1, -1)[0]
        cor = np.corrcoef(att_eff, np.linspace(0, 10, 11).astype(int))[0][1]
        return cor

    index_labels = ['Decreased F1-score', 'Decreased Entropy']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), subplot_kw=dict(polar=True))
    # Calculate angles for the radar chart
    labels = ['                    Relative Breadth', 'Congestion Level', 'Relative Depth                 ',
              '# of congested paths                         ', 'Attack Frequency']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # Repeat the first value to close the plot
    angles.append(angles[0])
    interval = (0.0, 0.1)
    for index in index_labels:
        if index == 'Decreased F1-score':
            cor_interval = interval_exp()
            cor_freq = freq_exp(interval)
            cor_rd = rd_exp(interval)
            cor_rb = rb_exp(interval)
            cor_con = conDegree_exp(interval)

            # 将列表中的数转换为极坐标下的坐标
            values = [cor_rb, cor_interval, cor_rd, cor_con, cor_freq]
        else:
            values = pd.read_csv(os.path.join(folderPath, 'entropy.csv')).to_numpy().T[0]

        # Create the radar chart
        values.append(values[0])
        # Plot the data
        ax.plot(angles, values, marker='o', linewidth=2, label=index)

        # Fill the area
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1], fontsize=10, style='italic')
        ax.set_xticklabels(labels, fontsize=10, style='italic')

        # Set the y-axis labels
        yticks = [-1, 0, 1]
        ax.set_yticks(yticks, fontsize=10, style='italic')
        ax.set_yticklabels(yticks, fontsize=10, style='italic')

    fig.legend(loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.95))
    # Show the plot
    fig.savefig(os.path.join(figure_savePath, 'fig9.png'), format='png', dpi=500, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    data_path = os.path.join(get_root_path(), 'examples', 'statistical_analysis')
    figure_savePath = os.path.join(get_root_path(), 'examples', 'figures')
    os.makedirs(figure_savePath, exist_ok=True)
    plotList = [f'fig{index}' for index in [2, 3, '4_a', '4_b', 5, 6, 7, 8, 9]]
    qbar = tqdm(plotList)
    for index in qbar:
        qbar.set_description(f"Generating figures")
        qbar.set_postfix_str(f"Drawing {index.capitalize()} ... ")
        locals()[f"draw_{index}"]()
