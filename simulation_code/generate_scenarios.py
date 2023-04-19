# 完全版
import h5py
import pandas as pd
import copy as cp
from joblib import Parallel, delayed

from get_link_congestion_pr_vector import load
from clink_simulator import *
from alg_scfs import alg_SCFS
from alg_clink import clink_algorithm, con_rm_gen
from alg_map import alg_map
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


def diagnosis(routing_matrix, linkstatus, pathstatus, pro, save_path, filename, freq=None, j=None):
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

    # 查看是否已有原文件
    status = None  # 只有status=1使用的刚生成的数据，其余都需要重新根据链路状态生成

    linkstatus_scfs_eval = None
    linkstatus_clink_eval = None
    linkstatus_map_eval = None
    try:
        alg_set = ['scfs', 'clink', 'map']
        linkstatus_scfs_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-scfs-high-eval.csv')).to_numpy()[:, 1:]
        linkstatus_clink_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-clink-high-eval.csv')).to_numpy()[:, 1:]
        linkstatus_map_eval = \
            pd.read_csv(os.path.join(save_path, f'{filename}-map-high-eval.csv')).to_numpy()[:, 1:]
        status = 0  # 已有
    except Exception:
        try:
            if 'attack' in save_path:
                # 查看有没有基础评估
                flag = 0
                for file in os.listdir(save_path):
                    if f'{freq}-{j + 1}-map-eval' in file:
                        flag = 1
                # 有基础评估
                if flag == 1:
                    with h5py.File(os.path.join(save_path, f"{filename}-scfs-eval.hdf5"), 'r') as f:
                        linkstatus_scfs_basic_eval = f['TP'][()], f['FP'][()], f['TN'][()], f['FN'][()]
                    with h5py.File(os.path.join(save_path, f"{filename}-clink-eval.hdf5"), 'r') as f:
                        linkstatus_clink_basic_eval = f['TP'][()], f['FP'][()], f['TN'][()], f['FN'][()]
                    with h5py.File(os.path.join(save_path, f"{filename}-map-eval.hdf5"), 'r') as f:
                        linkstatus_map_basic_eval = f['TP'][()], f['FP'][()], f['TN'][()], f['FN'][()]
                    status = 0  # 已有
                # 无基础评估
                else:
                    # 查看有无攻击数据
                    flag = 0
                    for file in os.listdir(save_path):
                        if '0.9-100-map' in file:
                            flag = 1
                    # 有攻击数据
                    if flag == 1:
                        with h5py.File(os.path.join(save_path, f"{filename}-scfs.hdf5"), 'r') as f:
                            linkstatus_scfs = f['linkstatus_attacked_scfs'][()]
                        with h5py.File(os.path.join(save_path, f"{filename}-clink.hdf5"), 'r') as f:
                            linkstatus_clink = f['linkstatus_attacked_clink'][()]
                        with h5py.File(os.path.join(save_path, f"{filename}-map.hdf5"), 'r') as f:
                            linkstatus_map = f['linkstatus_attacked_map'][()]
                        status = 0  # 已有
                    # 无攻击数据
                    else:
                        # scfs诊断
                        linkstatus_scfs = alg_SCFS(routing_matrix, pathstatus)
                        linkstatus_scfs = idx2bin(linkstatus_scfs, routing_matrix.shape[1])
                        with h5py.File(os.path.join(save_path, f"{filename}-scfs.hdf5"), 'w') as f:
                            f.create_dataset("linkstatus_attacked_scfs", data=linkstatus_scfs)
                        # clink诊断
                        # # 生成路由矩阵
                        R_set = con_rm_gen(routing_matrix, pathstatus, False, None, False)
                        linkstatus_clink, _ = clink_algorithm(R_set, pro)
                        linkstatus_clink = idx2bin(linkstatus_clink, routing_matrix.shape[1])
                        with h5py.File(os.path.join(save_path, f"{filename}-clink.hdf5"), 'w') as f:
                            f.create_dataset("linkstatus_attacked_clink", data=linkstatus_clink)
                        # map诊断
                        linkstatus_map, _ = alg_map(pathstatus.transpose(), routing_matrix, pro)
                        linkstatus_map = idx2bin(linkstatus_map, routing_matrix.shape[1])
                        with h5py.File(os.path.join(save_path, f"{filename}-map.hdf5"), 'w') as f:
                            f.create_dataset("linkstatus_attacked_map", data=linkstatus_map)
                        status = 1  # 没有
                    # # 得到链路性能
                    linkstatus_scfs_basic_eval = get_conf_mx(linkstatus, linkstatus_scfs,
                                                             routing_matrix.shape[1], False)
                    linkstatus_clink_basic_eval = get_conf_mx(linkstatus, linkstatus_clink,
                                                              routing_matrix.shape[1], False)
                    linkstatus_map_basic_eval = get_conf_mx(linkstatus, linkstatus_map,
                                                            routing_matrix.shape[1], False)
            else:
                # 查看有无基础评估
                flag = 0
                for file in os.listdir(save_path):
                    if 'eval' in file:
                        flag = 1
                if flag == 1:
                    with h5py.File(os.path.join(save_path, f"{filename}_scfs_eval.hdf5"), 'r') as f:
                        linkstatus_scfs_basic_eval = f['linkstatus_scfs_eval'][()]
                    with h5py.File(os.path.join(save_path, f"{filename}_clink_eval.hdf5"), 'r') as f:
                        linkstatus_clink_basic_eval = f['linkstatus_clink_eval'][()]
                    with h5py.File(os.path.join(save_path, f"{filename}_map_eval.hdf5"), 'r') as f:
                        linkstatus_map_basic_eval = f['linkstatus_map_eval'][()]
                    status = 0  # 已有
                else:
                    # scfs诊断
                    linkstatus_scfs = alg_SCFS(routing_matrix, pathstatus)
                    linkstatus_scfs = idx2bin(linkstatus_scfs, routing_matrix.shape[1])
                    with h5py.File(os.path.join(save_path, f"{filename}-scfs.hdf5"), 'w') as f:
                        f.create_dataset("linkstatus_scfs", data=linkstatus_scfs)
                    # clink诊断
                    # # 生成路由矩阵
                    R_set = con_rm_gen(routing_matrix, pathstatus, False, None, False)
                    linkstatus_clink, _ = clink_algorithm(R_set, pro)
                    linkstatus_clink = idx2bin(linkstatus_clink, routing_matrix.shape[1])
                    with h5py.File(os.path.join(save_path, f"{filename}-clink.hdf5"), 'w') as f:
                        f.create_dataset("linkstatus_clink", data=linkstatus_clink)
                    # map诊断
                    linkstatus_map, _ = alg_map(pathstatus.transpose(), routing_matrix, pro)
                    linkstatus_map = idx2bin(linkstatus_map, routing_matrix.shape[1])
                    with h5py.File(os.path.join(save_path, f"{filename}-map.hdf5"), 'w') as f:
                        f.create_dataset("linkstatus_map", data=linkstatus_map)

                    # # 得到链路性能
                    linkstatus_scfs_basic_eval = get_conf_mx(linkstatus, linkstatus_scfs,
                                                             routing_matrix.shape[1], False)
                    linkstatus_clink_basic_eval = get_conf_mx(linkstatus, linkstatus_clink,
                                                              routing_matrix.shape[1], False)
                    linkstatus_map_basic_eval = get_conf_mx(linkstatus, linkstatus_map,
                                                            routing_matrix.shape[1], False)
                    status = 1

        except Exception as e:
            # scfs诊断
            linkstatus_scfs = alg_SCFS(routing_matrix, pathstatus)
            linkstatus_scfs = idx2bin(linkstatus_scfs, routing_matrix.shape[1])
            with h5py.File(os.path.join(save_path, f"{filename}-scfs.hdf5"), 'w') as f:
                f.create_dataset("linkstatus_clink", data=linkstatus_scfs)
            # clink诊断
            # # 生成路由矩阵
            R_set = con_rm_gen(routing_matrix, pathstatus, False, None, False)
            linkstatus_clink, _ = clink_algorithm(R_set, pro)
            linkstatus_clink = idx2bin(linkstatus_clink, routing_matrix.shape[1])
            with h5py.File(os.path.join(save_path, f"{filename}-clink.hdf5"), 'w') as f:
                f.create_dataset("linkstatus_clink", data=linkstatus_clink)
            # map诊断
            linkstatus_map, _ = alg_map(pathstatus.transpose(), routing_matrix, pro)
            linkstatus_map = idx2bin(linkstatus_map, routing_matrix.shape[1])
            with h5py.File(os.path.join(save_path, f"{filename}-map.hdf5"), 'w') as f:
                f.create_dataset("linkstatus_clink", data=linkstatus_map)

            # 存储
            linkstatus_scfs_basic_eval = get_conf_mx(linkstatus, linkstatus_scfs,
                                                     routing_matrix.shape[1], False)
            linkstatus_clink_basic_eval = get_conf_mx(linkstatus, linkstatus_clink,
                                                      routing_matrix.shape[1], False)
            linkstatus_map_basic_eval = get_conf_mx(linkstatus, linkstatus_map,
                                                    routing_matrix.shape[1], False)
            status = 1

        linkstatus_scfs_eval = get_score(linkstatus_scfs_basic_eval[0], linkstatus_scfs_basic_eval[1],
                                         linkstatus_scfs_basic_eval[2], linkstatus_scfs_basic_eval[3])
        linkstatus_clink_eval = get_score(linkstatus_clink_basic_eval[0], linkstatus_clink_basic_eval[1],
                                          linkstatus_clink_basic_eval[2], linkstatus_clink_basic_eval[3])
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
    if linkstatus_scfs_eval is None or linkstatus_clink_eval is None or linkstatus_map_eval is None:
        raise ValueError("One or more evaluation variables were not assigned a value.")

    return np.stack([linkstatus_scfs_eval.astype(float),
                     linkstatus_clink_eval.astype(float),
                     linkstatus_map_eval.astype(float)]), status


def handle_sec_i(link_congestion_pro_vector, env, section, i, routing_matrix, dirs, rm_idx):
    # 依据先验概率重复生成1000个时刻的链路状态
    times = 1000
    root_path = os.path.join(get_root_path(), 'data', dirs[rm_idx].split('.')[0], env[1], '{}-{}-{}'
                             .format(section, i + 1, times))
    # If not exist, start to generate
    if not os.path.exists(os.path.join(root_path, f"{section}-{i + 1}-#1000-linkstatus_true.hdf5")):
        pro = link_congestion_pro_vector[env[0], section, i]

        linkstatus_true = (np.random.rand(times, routing_matrix.shape[1]) < pro) * 1
        pathstatus_true = path_stat_gen(linkstatus_true, routing_matrix)

        pathNum = routing_matrix.shape[0]
        for path in range(1, pathNum + 1):
            # Create the directories
            if not os.path.exists(os.path.join(root_path, 'attack', 'freq-topic', 'path', 'path-{}'.format(path))):
                os.makedirs(os.path.join(root_path, 'attack', 'freq-topic', 'path', 'path-{}'.format(path)))

        # scfs诊断
        linkstatus_scfs = alg_SCFS(routing_matrix, pathstatus_true, queue=None)
        linkstatus_scfs = idx2bin(linkstatus_scfs, routing_matrix.shape[1])
        # clink诊断
        # # 生成拥塞路由矩阵
        R_set = con_rm_gen(routing_matrix, pathstatus_true, False, None, False)
        linkstatus_clink, _ = clink_algorithm(R_set, pro)
        linkstatus_clink = idx2bin(linkstatus_clink, routing_matrix.shape[1])
        # map诊断
        linkstatus_map, _ = alg_map(pathstatus_true.transpose(), routing_matrix, pro)
        linkstatus_map = idx2bin(linkstatus_map, routing_matrix.shape[1])
        kinds = ['linkstatus_true', 'pathstatus_true', 'linkstatus_scfs', 'linkstatus_clink', 'linkstatus_map']
        files_name = ['{}-{}-#{}-{}.hdf5'.format(section, i + 1, times, kind) for kind in kinds]
        for idx, file_name in enumerate(files_name):
            with h5py.File(os.path.join(root_path, file_name), 'w') as f:
                f.create_dataset(kinds[idx], data=locals()[kinds[idx]])


def handle_sec_i_path(section, i, target_path, freq, j, routing_matrix, rm_idx, dirs, env, link_congestion_pro_vector):
    times = 1000
    root_path = os.path.join(get_root_path(), 'data', dirs[rm_idx].split('.')[0], env[1], '{}-{}-{}'
                             .format(section, i + 1, times))

    pro = link_congestion_pro_vector[env[0], section, i]

    att_path = os.path.join(root_path, 'attack', 'freq-topic', 'path', 'path-{}'.format(target_path))

    # 依据先验概率重复生成1000个时刻的链路状态
    truepath_name = '{}-{}-#{}-pathstatus_true.hdf5'.format(section, i + 1, times)
    truepath_path = os.path.join(root_path, truepath_name)
    with h5py.File(truepath_path, 'r') as f:
        pathstatus_true = f['pathstatus_true'][()]
    truelink_name = '{}-{}-#{}-linkstatus_true.hdf5'.format(section, i + 1, times)
    truelink_path = os.path.join(root_path, truelink_name)
    with h5py.File(truelink_path, 'r') as f:
        linkstatus_true = f['linkstatus_true'][()]
    diagnosis(routing_matrix, linkstatus_true, pathstatus_true, pro, root_path,
              f"{section}-{i + 1}-#1000-linkstatus", )

    loc_filename = '{}-{}-#{}-pathstatus-attacked-loc-{}-{}.csv'.format(section, i + 1, times, freq, j + 1)
    loc = np.random.choice(range(times), int(freq * times), replace=False)

    path_attacked_fileName = '{}-{}-#1000-pathstatus-attacked-{}-{}.csv'.format(section, i + 1, freq, j + 1)
    pathstatus_attacked = cp.deepcopy(pathstatus_true)
    pathstatus_attacked[loc, target_path - 1] = 1 - pathstatus_true[
        loc, target_path - 1]  # 路径观测值翻转
    # 存储信息
    df = pd.DataFrame(pathstatus_attacked)
    df.to_csv(os.path.join(att_path, path_attacked_fileName), index=False)
    df = pd.DataFrame(loc)
    df.to_csv(os.path.join(att_path, loc_filename), index=False)

    filename = f"{section}-{i + 1}-#1000-linkstatus-attacked-{freq}-{j + 1}"
    _, if_loc_need_regen = diagnosis(routing_matrix, linkstatus_true, pathstatus_attacked,
                                     pro, att_path, filename, freq, j)
    # 如果原目录有已有的诊断数据，那么生成的攻击路径状态文件和实际的链路诊断不符
    if if_loc_need_regen == 0:
        with h5py.File(os.path.join(att_path, f"{filename}-scfs.hdf5"), 'r') as f:
            linkstatus_scfs = f['linkstatus_attacked_scfs'][()]
        pathstatus_attacked = np.array(
            [np.dot(routing_matrix, linkstatus) for linkstatus in linkstatus_scfs])
        loc = np.unique(np.where(np.abs(pathstatus_attacked - pathstatus_true) == 1)[0])
        # 存储信息
        df = pd.DataFrame(pathstatus_attacked)
        df.to_csv(os.path.join(att_path, path_attacked_fileName), index=False)
        df = pd.DataFrame(loc)
        df.to_csv(os.path.join(att_path, loc_filename), index=False)


def scn_gen():
    a, b = ('场景-1: 不变方差', 'even'), ('场景-2: 可变方差', 'uneven')
    for env in [a, b]:
        # 真实场景，依据先验概率和路由矩阵，生成链路状态和路径状态
        if env == a:
            sections = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        else:
            sections = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
        topo_path = os.path.join(get_root_path(), 'datasets', 'tree_topo')
        dirs = os.listdir(topo_path)
        rm_set = [pd.read_csv(os.path.join(topo_path, dirs[i]), header=None).to_numpy() for i in range(len(dirs))]
        n = 100
        for rm_idx, routing_matrix in enumerate(rm_set):
            linkNum = routing_matrix.shape[1]
            link_congestion_pro_vector = load(linkNum, f'link_congestion_pro_vector_{linkNum}.pickle')
            path = list(range(1, routing_matrix.shape[0] + 1))  # 需要遍历的路径
            # 真实场景
            Parallel(n_jobs=-1)(
                delayed(handle_sec_i)(link_congestion_pro_vector, env, section, i, routing_matrix, dirs, rm_idx)
                for section in tqdm(sections, desc=f"Generate {dirs[rm_idx].split('.')[0]} no-attack scenarios")
                for i in range(n))
            # 攻击场景
            m = 100  # repeat attack times
            tu = [(freq, j) for freq in np.linspace(0.1, 0.9, 9) for j in range(m)]
            tu.append((1.0, 0))  # As when freq==1.0, there's no necessary to repeat
            Parallel(n_jobs=-1)(delayed(handle_sec_i_path)(section, i, target_path, freq, j, routing_matrix, rm_idx,
                                                           dirs, env, link_congestion_pro_vector)
                                for section in tqdm(sections, desc=f"Generate {dirs[rm_idx].split('.')[0]}"
                                                                   f" Byzantine-attack scenarios")
                                for i in range(n) for target_path in path for freq, j in tu)


if __name__ == '__main__':
    scn_gen()
