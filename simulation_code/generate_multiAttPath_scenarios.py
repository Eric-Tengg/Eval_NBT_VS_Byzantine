import h5py
import pandas as pd
import copy as cp

from get_link_congestion_pr_vector import load
from clink_simulator import *
from scfs import alg_SCFS
from clink import clink_algorithm, con_rm_gen
from alg_map import alg_map
from joblib import Parallel, delayed
from itertools import combinations
from evaluate import get_conf_mx


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def idx2bin(idx, num):
    # 输入多维的序号集， 返回多维二进制数组, num为链路数
    bin = np.zeros((idx.shape[0], num), dtype=int)
    for i, arr in enumerate(idx):
        if arr.shape[0]:
            bin[i, arr - 1] = 1
    return bin


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


def diagnosis(routing_matrix, linkstatus, pathstatus, pro, save_path, filename):
    if not os.path.exists(os.path.join(save_path, 'linkstatus_scfs_eval_' + filename + '.csv')):
        # scfs诊断
        linkstatus_scfs = alg_SCFS(routing_matrix, pathstatus)
        linkstatus_scfs = idx2bin(linkstatus_scfs, routing_matrix.shape[1])
        # # 得到链路性能
        linkstatus_scfs_basic_eval = get_conf_mx(linkstatus, linkstatus_scfs, routing_matrix.shape[1], False)
        linkstatus_scfs_eval = get_score(linkstatus_scfs_basic_eval[0], linkstatus_scfs_basic_eval[1],
                                         linkstatus_scfs_basic_eval[2], linkstatus_scfs_basic_eval[3])
        # clink诊断
        # # 生成路由矩阵
        R_set = con_rm_gen(routing_matrix, pathstatus, False, None, False)
        linkstatus_clink, _ = clink_algorithm(R_set, pro)
        linkstatus_clink = idx2bin(linkstatus_clink, routing_matrix.shape[1])
        # 得到链路性能
        linkstatus_clink_basic_eval = get_conf_mx(linkstatus, linkstatus_clink, routing_matrix.shape[1], False)
        linkstatus_clink_eval = get_score(linkstatus_clink_basic_eval[0], linkstatus_clink_basic_eval[1],
                                          linkstatus_clink_basic_eval[2], linkstatus_clink_basic_eval[3])
        # map诊断
        linkstatus_map, _ = alg_map(pathstatus.transpose(), routing_matrix, pro)
        linkstatus_map = idx2bin(linkstatus_map, routing_matrix.shape[1])
        # # 得到链路性能
        linkstatus_map_basic_eval = get_conf_mx(linkstatus, linkstatus_map, routing_matrix.shape[1], False)
        linkstatus_map_eval = get_score(linkstatus_map_basic_eval[0], linkstatus_map_basic_eval[1],
                                        linkstatus_map_basic_eval[2], linkstatus_map_basic_eval[3])
        alg_set = ['scfs', 'clink', 'map']
        for alg in alg_set:
            basic_eval = locals()['linkstatus_' + alg + '_basic_eval']
            pd.DataFrame(basic_eval, index=['TP', 'FP', 'TN', 'FN']).to_csv(
                os.path.join(save_path, f'{filename}-{alg}-basic-eval.csv'))

        for alg in alg_set:
            high_eval = locals()['linkstatus_' + alg + '_eval']
            pd.DataFrame(high_eval, index=['DR', 'FPR', 'F1']).to_csv(
                os.path.join(save_path, f'{filename}-{alg}-high-eval.csv'))
    else:
        print(f'已有{filename}数据')
        alg_set = ['scfs', 'clink', 'map']
        linkstatus_scfs_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-scfs-high-eval.csv')).to_numpy()[:, 1:]
        linkstatus_clink_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-clink-high-eval.csv')).to_numpy()[:, 1:]
        linkstatus_map_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-map-high-eval.csv')).to_numpy()[:, 1:]

    return np.stack([linkstatus_scfs_eval.astype(float),
                     linkstatus_clink_eval.astype(float),
                     linkstatus_map_eval.astype(float)])


def handle_m_j(pathAtt, dirs, rm_idx, env, section, times, i, routing_matrix, pro):
    # # 得到攻击的路径集
    # 攻击路径数
    m = pathAtt.shape[0]
    # 文件路径设置
    root_path = os.path.join(get_root_path(), 'data', dirs[rm_idx].split('.')[0], env[1],
                             '{}-{}-{}'.format(section, i + 1, times))
    save_path = os.path.join(root_path, 'attack', 'freq-topic', 'path', f'path-{m}-{pathAtt}')

    os.makedirs(save_path, exist_ok=True)
    with h5py.File(os.path.join(root_path, '{}-{}-#{}-pathstatus_true.hdf5'.format(section, i + 1, times)), 'r') as f:
        pathstatus_true = f['pathstatus_true'][()]
    with h5py.File(os.path.join(root_path, '{}-{}-#{}-linkstatus_true.hdf5'.format(section, i + 1, times)), 'r') as f:
        linkstatus_true = f['linkstatus_true'][()]
    # 重复选取时刻20次
    for k in range(20):
        loc_filename = '{}-{}-#1000-pathstatus-attacked-loc-{}.csv'.format(section, i + 1, k + 1)
        loc = np.random.choice(range(times), int(times / m), replace=False)
        # 存储信息
        pd.DataFrame(loc).to_csv(os.path.join(save_path, loc_filename), index=False)

        pathstatus_attacked = cp.deepcopy(pathstatus_true)
        for path in pathAtt:
            pathstatus_attacked[loc, path - 1] = 1 - pathstatus_true[loc, path - 1]  # 路径观测值翻转

        path_attacked_fileName = '{}-{}-#1000-pathstatus-attacked-{}.csv'.format(section, i + 1, k + 1)

        # 存储信息
        pd.DataFrame(pathstatus_attacked).to_csv(os.path.join(save_path, path_attacked_fileName), index=False)

        filename = f'{section}-{i + 1}-#1000-linkstatus-attacked-{k + 1}'
        diagnosis(routing_matrix, linkstatus_true, pathstatus_attacked, pro, save_path, filename)


def multi_topo_att_scn_gen_breadth():
    a, b = ('场景-1: 不变方差', 'even'), ('场景-2: 可变方差', 'uneven')
    env = a
    # 真实场景，依据先验概率和路由矩阵，生成链路状态和路径状态
    if env == a:
        sections = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    else:
        sections = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    dirs = os.listdir(os.path.join(get_root_path(), 'datasets', 'tree_topo'))
    rm_set = [pd.read_csv(os.path.join(get_root_path(), 'datasets', 'tree_topo', dirs[i]), header=None).to_numpy()
              for i in range(len(dirs))]
    for rm_idx, routing_matrix in enumerate(rm_set):
        linkNum = routing_matrix.shape[1]
        pathSet = list(range(1, routing_matrix.shape[0] + 1))
        link_congestion_pro_vector = load(linkNum, f'link_congestion_pro_vector_{linkNum}.pickle')
        for section in sections:  # 概率区间值
            for i in range(20):
                # 攻击环境
                pro = link_congestion_pro_vector[env[0], section, i]
                times = 1000  # 攻击 m 条路径, 攻击次数为1000，攻击次数平分给每条路径
                # m 是攻击路径数，j 是重复筛选次数
                # 对所有情况进行实验，直接传入攻击路径
                pathAttSet_all = []
                for m in range(2, routing_matrix.shape[0] + 1):
                    pathAttSet_all.append(list(combinations(pathSet, m)))
                pathAttSet_all = [np.array(item) for sublist in pathAttSet_all for item in sublist]
                # 筛除已经跑过的攻击路径
                pathAttSet = []
                for pathAtt in pathAttSet_all:
                    if os.path.exists(
                            os.path.join(get_root_path(), 'data', dirs[rm_idx].split('.')[0], env[1], '{}-{}-{}'
                                    .format(section, i + 1, times), 'attack', 'freq-topic', 'path',
                                         f'path-{pathAtt.shape[0]}-{pathAtt}',
                                         f"{section}-{i + 1}-#1000-linkstatus-attacked-20-clink-high-eval.csv")):
                        print(f"{dirs[rm_idx].split('.')[0]} 拓扑 - {pathAtt} 路径集已完成攻击")
                    else:
                        pathAttSet.append(pathAtt)

                # 多进程
                Parallel(n_jobs=-1)(
                    delayed(handle_m_j)(pathAtt, dirs, rm_idx, env, section, times, i, routing_matrix, pro)
                    for pathAtt in tqdm(pathAttSet, desc=f"{section}-{i + 1}"))


if __name__ == '__main__':
    multi_topo_att_scn_gen_breadth()
