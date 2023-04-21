import h5py
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def get_conf_mx(links_state, links_state_inferred, num, is_idx=True):
    TP = np.zeros(links_state.shape[0], dtype=int)
    FP = np.zeros(links_state.shape[0], dtype=int)
    TN = np.zeros(links_state.shape[0], dtype=int)
    FN = np.zeros(links_state.shape[0], dtype=int)
    if is_idx:
        for i in range(links_state.shape[0]):
            link_stat = links_state[i].astype(int)
            link_stat_inf = links_state_inferred[i].astype(int)
            TP[i] = len(np.intersect1d(link_stat, link_stat_inf))
            FP[i] = len(np.setdiff1d(link_stat_inf, link_stat))
            FN[i] = len(np.setdiff1d(link_stat, link_stat_inf))
            TN[i] = num - TP[i] - FP[i] - FN[i]
    else:
        for i in range(links_state.shape[0]):
            link_stat = links_state[i].astype(int)
            link_stat_inf = links_state_inferred[i].astype(int)
            conf_mx = confusion_matrix(link_stat, link_stat_inf).ravel()
            if conf_mx.shape[0] != 1:
                TN[i], FP[i], FN[i], TP[i] = conf_mx
            else:
                if np.all(link_stat):
                    TN[i], FP[i], FN[i], TP[i] = 0, 0, 0, num
                else:
                    TN[i], FP[i], FN[i], TP[i] = num, 0, 0, 0
    return TP, FP, TN, FN


def get_score(tp, fp, tn, fn):
    P = tp + fn  # 真实
    N = tn + fp
    X = tp + fp  # 预测
    env1 = np.where((P == 0) & (X == 0))[0]  # 全为正常链路, 且预测正确
    env2 = np.where((P == 0) & (X != 0))[0]  # 全为正常链路, 且预测错误
    env3 = np.where(N == 0)[0]  # 全为拥塞链路
    tp[env1], fn[env1] = 1, 0  # DR=1
    tp[env2], fn[env2] = 0, 1  # DR=0
    tn[env3], fp[env3], tp[env3] = 1, 0, 1

    DR_times = tp / (tp + fn)  # TP / P
    FPR_times = fp / (tn + fp)  # FP / N
    F1_times = tp / (tp + 0.5 * (fp + fn))
    F1_times[np.where(np.isnan(F1_times) is True)[0]] = 1

    DR, FPR, F1 = DR_times, FPR_times, F1_times

    return np.stack([DR, FPR, F1])


def genFig3Data():
    """
    对不同链路类型 DR, FPR, F1 下降的概率统计及CDF图
    """
    topoName = '10-16'
    for scn in ['even', 'uneven']:
        folderPath = os.path.join(get_root_path(), 'data', topoName, scn)
        root_savePath = os.path.join(get_root_path(), 'analysis', 'statistical_analysis', 'fig3', '10-16', scn)
        sections = [(0, round(i + 0.1, 1)) for i in np.linspace(0.0, .9, 10).round(1)]
        path_set = np.linspace(1, 10, 10).astype(int)
        routing_matrix = pd.read_csv(os.path.join(get_root_path(), 'dataset', 'topo_breadth',
                                                  'topoSet4', f'{topoName}.csv'), header=None).to_numpy()
        for section in sections:
            for i in tqdm(range(20), desc=f"{section}"):
                # section-i/ true & path / j / {scfs & clink & map}.csv / root & intermediate & edge
                sub_savePath = os.path.join(root_savePath, f'{section}-{i + 1}')
                os.makedirs(os.path.join(sub_savePath, 'true'), exist_ok=True)
                sub_dataPath = os.path.join(folderPath, '{}-{}-1000'.format(section, i + 1))

                if not os.path.exists(os.path.join(sub_savePath, 'true', 'linkstatus_map_high_eval.csv')):
                    # 得到真实的链路数据(用于计算链路性能)和真实状态下的诊断(用于计算攻击导致性能的下降幅度)
                    # 得到 linkstatus_true, linkstatus_scfs, linkstatus_clink, linkstatus_map
                    for data_kind in ['true', 'scfs', 'clink', 'map']:
                        file_p = os.path.join(sub_dataPath, f'{section}-{i + 1}-#1000-linkstatus_{data_kind}.hdf5')
                        with h5py.File(file_p, 'r') as f:
                            locals()[f'linkstatus_{data_kind}'] = f[f'linkstatus_{data_kind}'][()].T  # row is linkIdx
                    # 计算真实状态下，每种链路在不同算法的一级性能评估(TP, FP, TN, FN)
                    # 得到 linkstatus_scfs_basic_eval, linkstatus_clink_basic_eval, linkstatus_map_basic_eval
                    for alg_kind in ['scfs', 'clink', 'map']:
                        locals()['linkstatus_' + alg_kind + '_basic_eval'] = \
                            get_conf_mx(locals()['linkstatus_true'], locals()['linkstatus_' + alg_kind], 1000, False)
                        basic_eval = locals()['linkstatus_' + alg_kind + '_basic_eval']
                        # 存储
                        df = pd.DataFrame(
                            {'TP': basic_eval[0], 'FP': basic_eval[1], 'TN': basic_eval[2], 'FN': basic_eval[3]})
                        df.to_csv(os.path.join(sub_savePath, 'true', 'linkstatus_' + alg_kind + '_basic_eval.csv'),
                                  index=False)
                    # 计算真实状态下，每种链路在不同算法的二/三级性能评估(DR, FPR, F1)
                    # 得到 linkstatus_scfs_high_eval, linkstatus_clink_high_eval, linkstatus_map_high_eval
                    for alg_kind in ['scfs', 'clink', 'map']:
                        basic_eval = locals()['linkstatus_' + alg_kind + '_basic_eval']
                        locals()['linkstatus_' + alg_kind + '_high_eval'] = \
                            get_score(basic_eval[0], basic_eval[1], basic_eval[2], basic_eval[3])
                        high_eval = locals()['linkstatus_' + alg_kind + '_high_eval']
                        # 存储
                        df = pd.DataFrame({'DR': high_eval[0], 'FPR': high_eval[1], 'F1': high_eval[2]})
                        df.to_csv(os.path.join(sub_savePath, 'true', F'linkstatus_{alg_kind}_high_eval.csv'),
                                  index=False)

                for path in path_set:
                    # path / j / {scfs & clink & map}.csv / root & intermediate & edge
                    sub2_dataPath = os.path.join(sub_dataPath, 'attack', 'freq-topic', 'path', f'path-{path}')
                    sub2_savePath = os.path.join(sub_savePath, f'path-{path}')
                    if not os.path.exists(os.path.join(sub2_savePath, f'linkstatus_attacked_map_high_eval.csv')):
                        # 得到攻击状态下的诊断
                        # 得到 linkstatus_attacked_scfs, linkstatus_attacked_clink, linkstatus_attacked_map
                        for alg_kind in ['scfs', 'clink', 'map']:
                            with h5py.File(os.path.join(sub2_dataPath, '{}-{}-#1000-linkstatus-attacked-1.0-1-{}.hdf5'
                                    .format(section, i + 1, alg_kind)), 'r') as f:
                                locals()['linkstatus_attacked_' + alg_kind] = f[f'linkstatus_attacked_{alg_kind}'][()].T
                        # 计算攻击状态下，每种链路在不同算法的一级性能评估(TP, FP, TN, FN)
                        # 得到 linkstatus_attacked_scfs_basic_eval,
                        #       linkstatus_attacked_clink_basic_eval, linkstatus_attacked_map_basic_eval
                        os.makedirs(os.path.join(sub2_savePath), exist_ok=True)
                        for alg_kind in ['scfs', 'clink', 'map']:
                            locals()['linkstatus_attacked_' + alg_kind + '_basic_eval'] = \
                                get_conf_mx(locals()['linkstatus_true'], locals()[f'linkstatus_attacked_{alg_kind}'],
                                            1000, False)
                            basic_eval = locals()[f'linkstatus_attacked_{alg_kind}_basic_eval']
                            # 存储
                            df = pd.DataFrame(
                                {'TP': basic_eval[0], 'FP': basic_eval[1], 'TN': basic_eval[2], 'FN': basic_eval[3]})
                            df.to_csv(
                                os.path.join(sub2_savePath, f'linkstatus_attacked_{alg_kind}_basic_eval.csv'),
                                index=False)
                        # 计算攻击状态下，每种链路在不同算法的二/三级性能评估(DR, FPR, F1)
                        # 得到 linkstatus_attacked_scfs_high_eval,
                        #       linkstatus_attacked_clink_high_eval, linkstatus_attacked_map_high_eval
                        for alg_kind in ['scfs', 'clink', 'map']:
                            basic_eval = locals()[f'linkstatus_attacked_{alg_kind}_basic_eval']
                            locals()[f'linkstatus_attacked_{alg_kind}_high_eval'] = \
                                get_score(basic_eval[0], basic_eval[1], basic_eval[2], basic_eval[3])
                            high_eval = locals()[f'linkstatus_attacked_{alg_kind}_high_eval']
                            # 存储
                            df = pd.DataFrame({'DR': high_eval[0], 'FPR': high_eval[1], 'F1': high_eval[2]})
                            df.to_csv(os.path.join(sub2_savePath, f'linkstatus_attacked_{alg_kind}_high_eval.csv'),
                                      index=False)
                    # 统计各种类型链路的综合评估
                    # ( sub2_savePath / {scfs & clink & map}_based_on_linkKinds.csv / root & intermediate & edge)
                    link_set = {'root': [1], 'intermediate': [2, 4, 7, 11, 13],
                                'edge': [3, 5, 6, 8, 9, 10, 12, 14, 15, 16]}

                    for alg_kind in ['scfs', 'clink', 'map']:
                        high_eval = pd.read_csv(
                            os.path.join(sub2_savePath, f'linkstatus_attacked_{alg_kind}_high_eval.csv')).to_numpy().T
                        high_eval_true = pd.read_csv(
                            os.path.join(sub_savePath, 'true', f'linkstatus_{alg_kind}_high_eval.csv')).to_numpy().T

                        locals()[f'{alg_kind}_based_on_linkKinds'] = {}
                        for link_kind in link_set:
                            linkSet = (np.array(link_set[link_kind]) - 1).tolist()
                            tmp = np.array([np.mean((high_eval_true[i] - high_eval[i])[linkSet]) for i in range(3)])
                            locals()[f'{alg_kind}_based_on_linkKinds'][link_kind] = [tmp[0], tmp[1] * -1, tmp[2]]

                        # 存储
                        df = pd.DataFrame(locals()[f'{alg_kind}_based_on_linkKinds'], index=['DR', 'FPR', 'F1'])
                        df.to_csv(os.path.join(sub2_savePath, f'{alg_kind}_based_on_linkKinds_relative.csv'))


genFig3Data()
