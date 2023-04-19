import math
import copy as cp
from multiprocessing import Pool, Pipe
import numpy as np
from tqdm import tqdm


def con_link_prob(X, link_prob_abs):
    pr = 1
    heal_link = np.where(X == 0)[0]
    for j in range(X.shape[0]):
        pr *= link_prob_abs[j] if j not in heal_link else 1 - link_prob_abs[j]
    return pr


ToBatch = lambda arr, size: [arr[i * size:(i + 1) * size] for i in range((size - 1 + len(arr)) // size)]


def batch_exec(f, args, w):
    args_batch = args[0]
    R = args[1]
    link_prob = args[2]
    results = np.zeros(0)
    for i, args in enumerate(args_batch):
        try:
            ans = f(args, R, link_prob)
            results = np.append(results, ans)
        except Exception as e:
            # print(e)
            results = np.append(results, None)
        w.send(1)
    return results


def multi_process_exec(f, args_mat, pool_size=5, desc=None):
    results = np.zeros(0)
    if len(args_mat['E_a']) == 0:
        return results
    batch_size = max(1, int(len(args_mat['E_a']) / 4 / pool_size))
    args_batches = ToBatch(args_mat['E_a'], batch_size)
    E_a = args_mat['E_a']
    with tqdm(total=len(args_mat['E_a']), desc=desc, postfix=f'正在处理链路集{E_a}') as pbar:
        with Pool(processes=pool_size) as pool:
            r, w = Pipe(duplex=False)
            pool_rets = []
            for i, args_batch in enumerate(args_batches):
                pool_rets.append(
                    pool.apply_async(batch_exec, (f, (args_batch, args_mat['R'], args_mat['link_prob']), w)))
            cnt = 0
            while cnt < len(args_mat['E_a']):
                try:
                    msg = r.recv()
                    pbar.update(1)
                    cnt += 1
                except EOFError:
                    break
            for ret in pool_rets:
                try:
                    for r in ret.get(timeout=1):
                        results = np.append(results, r)
                except TimeoutError:
                    print("TimeoutError caught for task")
    return results


def posteriori_com(e_k, R, link_prob):
    # print(e_k, R, link_prob)
    # 筛选出Domain(e_k)，得到e_k拥塞导致{路径集}拥塞的概率
    max_bt = R.copy()
    max_bt = max_bt[np.where(max_bt == e_k)[0]]
    # # 去除冗余列
    red_rank = np.where(np.array([np.any(max_bt[:, i]) for i in range(max_bt.shape[1])]) == False)[0]
    max_bt = np.delete(max_bt, red_rank, axis=1)
    # 得到该删减路由中的链路集
    E_a = np.unique(max_bt)
    E_a = np.delete(E_a, np.where(E_a == 0)[0])

    # 遍历所有的链路情况——二进制表示(0——正常，1——拥塞)
    X_set = np.zeros(0)
    for X in range(1, pow(2, E_a.shape[0])):
        X = np.array(list(bin(X)[2:]))
        diff = E_a.shape[0] - X.shape[0]
        X = np.append(np.array([0] * diff), X)
        X_set = np.append(X_set, X).astype(int).reshape(-1, E_a.shape[0])
    # 筛除出不能导致该坏子树拥塞的链路情况
    X_nonSence = np.array([index for index, X in enumerate(X_set) for i in range(max_bt.shape[0])
                           if np.dot(X, max_bt[i]) == 0]).astype(int)
    X_set = np.delete(X_set, X_nonSence, axis=0)

    # 计算所有可能出现的链路情况概率
    link_prob_abs = link_prob.copy()[E_a - 1]
    pr = np.array([con_link_prob(X, link_prob_abs) for X in X_set])
    # # 求和
    pr_sum = np.sum(pr)

    # 目标链路拥塞
    idx = np.where(E_a == e_k)[0][0]  # 目标链路在E_a中对应位置
    idx_X = [index for index, X in enumerate(X_set) if X[idx] == 1]
    # # 求和
    pr_k = np.array([pr[i] for i in idx_X])
    pr_k_sum = np.sum(pr_k)

    # 计算条件概率
    map_k = pr_k_sum / pr_sum
    return map_k


def foreach_Ea(E_a, R, link_prob):
    max_prob, e_k, map_o = 0, 0, 0
    for e_o in E_a:
        map_o = posteriori_com(e_o, R, link_prob)
        if map_o > max_prob:
            max_prob = map_o
            e_k = e_o
    return e_k, max_prob


def algorithm_map(R_set, link_prob):
    def update_R(R, ek_s):
        if type(ek_s) is not np.ndarray:
            ek_s = np.array([ek_s])
        for e_k in ek_s:
            del_shape = np.where(R == e_k)
            R = np.delete(R, del_shape[1], axis=1)
            R = np.delete(R, del_shape[0], axis=0)
        return R

    known_set = {}
    X_set = np.zeros(R_set.shape[0], dtype=object)
    for idx, R in enumerate(R_set):
        R_copy = cp.deepcopy(R)
        if R.shape[0] == R.shape[1]:
            X_set[idx] = np.delete(np.unique(R), np.where(np.unique(R) == 0)[0])
            continue
        map_set = np.zeros(0)
        defi_con_link_idx = [i for i in range(R.shape[0]) if np.where(R[i, :] != 0)[0].shape[0] == 1]  # 一定拥塞的链路
        X = np.delete(np.unique(R[defi_con_link_idx]), np.where(np.unique(R[defi_con_link_idx]) == 0))
        R = update_R(R, X)
        while R.shape[0] != 0:
            E_a = np.delete(np.unique(R), np.where(np.unique(R) == 0)[0])
            if str(E_a) in known_set:
                X_set[idx] = np.append(X, known_set[str(E_a)])
                continue
            else:
                e_k, max_prob = foreach_Ea(E_a, R, link_prob)
                X = np.append(X, e_k)
                map_set = np.append(map_set, max_prob)
                R = update_R(R, e_k)
        known_set[str(R_copy)] = X
        X_set[idx] = X
    return X_set


def clink_algorithm(reduced_rm, link_p, queue=None):
    def get_domain(reduced_congested_rm, link_probability):
        # 计算Domain值
        Domain = np.zeros(link_probability.shape[0], dtype=int)
        for i in range(link_probability.shape[0]):
            Domain[i] = np.where(reduced_congested_rm == i + 1)[0].shape[0]
        return Domain

    def updated_qb_conrm(congested_rm, e_k, q_b, reduced_congested_rm):
        # 更新Q_b
        resolved_path = np.where(congested_rm == e_k)[0] + [1]
        new_Q_b = np.zeros(0, dtype=int)
        for i in q_b:
            if i not in resolved_path:
                new_Q_b = np.append(new_Q_b, i)
        q_b = new_Q_b.copy()
        # 更新拥塞路由矩阵
        del_shape = np.where(reduced_congested_rm == e_k)
        # # 去除的部分
        del_part = reduced_congested_rm[del_shape[0]]
        # # 删减拥塞路由矩阵的行
        reduced_congested_rm = np.delete(reduced_congested_rm, del_shape[0], 0)

        return q_b, reduced_congested_rm, del_part

    def get_cur_link(reduced_congested_rm):
        # # 得到当前存在链路
        rm_1d = np.unique(reduced_congested_rm.reshape(1, -1)[0])
        E_a = np.delete(rm_1d, np.where(rm_1d == 0)[0])
        return E_a

    def get_link_of_min_score(reduced_congested_rm, link_probability, Domain):
        # 得到分数最小链路
        min_score, e_k = 10000, 10000
        # # 得到当前存在链路
        E_a = get_cur_link(reduced_congested_rm)
        # # 计算各链路的后验分数
        for link in E_a:
            score = math.log2(((1 - link_probability[link - 1]) / link_probability[link - 1]) / math.log2(Domain[link - 1] + 1))
            if score < min_score:
                min_score = score
                e_k = link
        return e_k, min_score

    X = np.zeros(reduced_rm.shape[0], dtype=object)
    pr = np.zeros(reduced_rm.shape[0], dtype=float)
    for idx, reduced_rm_sin in enumerate(reduced_rm):
        # print("拥塞路由矩阵:\n", reduced_rm_sin)
        con_rm = reduced_rm_sin.copy()
        P_c = np.array(range(1, reduced_rm_sin.shape[0] + 1))  # 重编拥塞路径集
        X_sin = np.zeros(0, dtype=int)
        Q_b = P_c.copy()
        while Q_b.shape[0] != 0:
            # 计算Domain值
            Domain = get_domain(con_rm, link_p)

            # 得到分数最小链路
            e_k, min_score = get_link_of_min_score(con_rm, link_p, Domain)

            # Add e_k to the solution X
            X_sin = np.append(X_sin, e_k)

            # 更新Q_b和拥塞链路由矩阵
            Q_b, con_rm, _ = updated_qb_conrm(reduced_rm_sin, e_k, Q_b, con_rm)

        # X.append(idx_link_state(X_sin, link_p.shape[0]))
        X[idx] = np.sort(X_sin)
        pr_copy = cp.deepcopy(1 - link_p)
        pr_copy[X_sin - 1] = link_p[X_sin - 1]
        pr[idx] = np.product(pr_copy)

    if queue is None:
        return X, pr
    else:
        queue.put((X, pr))


def con_rm_gen(routine_matrix, path_status, is_tqdm=True, queue=None, ret_list=False):
    rm_idx = np.zeros(path_status.shape[0], dtype=object)
    if is_tqdm:
        bar = enumerate(tqdm(path_status, desc='生成拥塞路由矩阵'))
    else:
        bar = enumerate(path_status)
    for idx, path_status_sin in bar:
        # 为路由矩阵进行编号
        rm_idx_sin = routine_matrix.copy()
        for i in range(rm_idx_sin.shape[0]):
            for j in range(rm_idx_sin.shape[1]):
                rm_idx_sin[i][j] = j + 1 if rm_idx_sin[i][j] != 0 else 0
        # 记录健康链路
        heal_link = np.zeros(0, dtype=int)
        for heal_path_idx in np.where(path_status_sin == 0)[0]:
            heal_path = rm_idx_sin[heal_path_idx]
            heal_link = np.append(heal_link, np.where(heal_path != 0)[0])
        # 删除健康路径
        rm_idx_sin = np.delete(rm_idx_sin, np.where(path_status_sin != 1)[0], 0)
        # 删除健康链路
        rm_idx[idx] = np.delete(rm_idx_sin, heal_link, 1)
        if ret_list:
            rm_idx[idx] = rm_idx[idx].tolist()
    if ret_list:
        rm_idx = rm_idx.tolist()
    if queue is None:
        return rm_idx
    else:
        queue.put(rm_idx)


if __name__ == '__main__':
    # pass
    # link_prob = np.array([0.03, 0.17, 0.12])
    # print("拥塞链路概率:\n", link_prob)
    # A_rm = np.array([[1, 1, 0], [1, 0, 1]])
    # print("路由矩阵:\n", A_rm)
    # real_con_link = np.array([[0, 1, 1], [1, 0, 0]])
    # print("真实链路状态:\n", real_con_link)
    # path_stat = np.array([1, 1])
    # print("路径状态:\n", path_stat)

    # 例1
    path_stat = np.array([[0, 0, 1, 1]])

    A_rm = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1]])
    print("路由矩阵:\n", A_rm)
    #
    link_prob = np.array([0.02, 0.01, 0.03, 0.02, 0.1, 0.04, 0.3])
    print("链路拥塞概率:\n", link_prob)

    R = np.array([[3, 0, 5],
                  [3, 4, 0]])
    print(algorithm_map(R, link_prob))

    # 例2
    # path_stat = np.array([[1, 1]])
    # R = np.array([[1, 2, 0],
    #               [1, 0, 3]])
    # link_prob = np.array([0.15, 0.3, 0.12])

    # 得到拥塞路由矩阵
    # con_rm_idx = con_rm_gen(A_rm, path_stat)
    # # result_clink = clink_algorithm(con_rm_idx, link_prob)[0]
    # result_clink = clink_algorithm(con_rm_idx, link_prob)
    # # result_clink = result_clink
    # print("CLINK算法结果:\n", result_clink)

    # draw_topo(A_rm, result_clink, result_clink, path_stat[0])
    # detction(real_con_link, result_clink)
