# 链路先验概率的生成、生成链路状态、生成路径状态、得到拥塞路由矩阵
import os
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm


def prob_stat_gen(link_num: int,
                  prob_times: int,
                  stat_times: int,
                  mu: np.ndarray = np.array([0.05, 0.15, 0.25]),
                  sigma: float = 0.1 / 6):
    """
    生成(mu.Shape[0], prob_times, link_num)维度的链路先验拥塞概率
        和 (mu.Shape[0], prob_times, stat_times, link_num)维度的链路状况

    Parameters
    ----------
    mu : array_like of floats
        先验拥塞概率均值, mu = (max - min) / 2
        输入为 array 时，mu.Shape[0] = link_prob.Shape[0] = link_stat.Shape[0]
    sigma : float or array_like of floats
        单次生成先验概率的标准差, sigma = (max - min) / 6
        输入为 array 时，sigma.Shape[0] = mu.Shape[0]
    link_num : int
        单次生成的链路状态中链路的条数, link_num = link_prob.Shape[2] = link_stat.Shape[3]
    prob_times : int
        单个均值下生成的链路先验概率次数, prob_times = link_prob.Shape[1] = link_stat.Shape[1]
    stat_times : int
        以单次链路拥塞概率生成的链路状态次数, stat_times = link_stat.Shape[2]

    Returns
    -------
    link_prob : ndarray
        链路先验拥塞概率集, link_prob.Shape = (mu.Shape[0], prob_times, link_num)
    link_stat : ndarray
        链路状态集, link_stat.Shape = (mu.Shape[0], prob_times, stat_times, link_num)

    Notes
    -----
    概率服从高斯分布
        math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                             e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },


    e.g.
    link_prob, link_stat = prob_stat_gen(mu=array[0.1, 0.2], sigma=0.1/6, link_num=3, prob_times=2, stat_times=3)

    link_prob = array[[[0.0937902  0.12274236 0.08778597]
                    [0.11283505 0.11471121 0.09881397]]
                    [[0.19346289 0.20085128 0.21693804]
                    [0.21399742 0.19257737 0.22474646]]]
    link_stat = array[[[[0 0 0]   [[0 0 1]      [[[0 1 0]       [[0 1 0]
                       [1 0 0]     [0 0 0]       [1 0 0]        [1 1 0]
                      [0 0 0]],   [0 1 0]]],    [1 0 0]],      [1 0 0]]]]
    """
    prob_shape = (link_num, prob_times, mu.shape[0])
    link_prob = np.random.normal(mu, sigma, prob_shape).transpose()
    link_prob[np.where(link_prob < 0)] = mu[np.where(link_prob < 0)[0]]  # 将小于0的概率改为该点所处区域的均值
    link_prob[np.where(link_prob >= 1)] = mu[np.where(link_prob >= 1)[0]]  # 将大于1的概率改为该点所处区域的均值
    link_stat = np.zeros((mu.shape[0], prob_times, stat_times, link_num), dtype=int)
    for i in tqdm(range(mu.shape[0]), desc=f"生成链路概率"):
        for j in range(prob_times):
            link_stat[i][j] = np.random.rand(stat_times, link_num) < link_prob[i][j]
    return link_prob, link_stat


def link_prob_gen(num):
    # 链路先验概率的生成
    # 拥塞概率范围[0.01, 0.3)
    link_prob = np.round(np.random.uniform(0.01, 0.3, num), 2)
    return link_prob


def link_stat_gen(link_probability, link_number):
    # 生成链路状态
    # 通过 各链路的拥塞概率，模拟生成 多个状态的 各链路的性能状态
    link_status = np.zeros(0, dtype=int)
    for i in range(link_number):
        sin_stat = (np.random.random(link_probability.shape[0]) < link_probability) * 1
        link_status = np.append(link_status, sin_stat)
    return link_status.reshape(link_number, -1)


def batch_exec(link_status, routine_matrix):
    path_status = np.zeros(0, dtype=int)
    for i in range(link_status.shape[0]):
        for j in range(routine_matrix.shape[0]):
            sin_path_status = (np.dot(link_status[i], routine_matrix[j]) > 0) * 1
            path_status = np.append(path_status, sin_path_status)
    return path_status


def path_stat_gen(link_status, routing_matrix):
    """
    return path_status.reshape(link_status.shape[0], -1)
    """
    # 生成路径状态
    # 由 拓扑结构 和 链路性能 生成 路径的性能状态
    path_status = np.zeros(0)
    pool_size = os.cpu_count()
    batch_size = int(link_status.shape[0] / 10 / pool_size)
    link_status_batch_set = np.array([np.array(link_status[i * batch_size: (i + 1) * batch_size])
                                      for i in range(int(link_status.shape[0] / batch_size))])
    pool = Pool(pool_size)
    ret = []
    for link_status_batch in link_status_batch_set:
        ret.append(pool.apply_async(batch_exec, args=(link_status_batch, routing_matrix)))
    for r in tqdm(ret, desc='生成路径状态'):
        path_status = np.append(path_status, r.get())
    if link_status.shape[0] % batch_size:
        target = batch_size * link_status_batch_set.shape[0]
        path_status = np.append(path_status,
                                batch_exec(link_status[target: link_status.shape[0]], routing_matrix))
    return path_status.reshape(link_status.shape[0], -1).astype(int)


def sim_rm(A_rm: np.ndarray, path_stat: np.ndarray):
    # 得到拥塞路由矩阵
    # 随机生成路径状态
    leafNodes = []
    for i in range(A_rm.shape[1]):
        if sum(A_rm[:, i]) == 1:
            leafNodes.append(i + 1)
    # print("leaf:\t", leafNodes)
    # def con_rm(routine_matrix, path_stat):
    # 状态与路由矩阵进行乘积
    con_rm = []
    heal_link_total = []
    for j in range(path_stat.shape[0]):
        heal_path = A_rm[np.where(path_stat[j] == 0)]  # 好路径的位置
        con_path = A_rm[np.where(path_stat[j] == 1)]  # 拥塞路径的位置
        # print("正常路径:\n", heal_path)
        # print("拥塞路径:\n", con_path)
        heal_link = np.zeros(0, dtype=int)
        for i in range(heal_path.shape[0]):
            heal_link_sin = np.where(heal_path[i] == 1)[0]
            # print(heal_link_sin)
            for link in heal_link_sin:
                if link not in heal_link:
                    heal_link = np.append(heal_link, link)
            # print(np.sort(heal_link))
        heal_link = np.sort(heal_link)
        heal_link_total.append(heal_link)
        # print("好链路",heal_link)
        # print("好路径",heal_path)
        # print("总的好链路",heal_link_total)
        # print("正常链路:\n", heal_link)
        sin_con_rm = np.zeros(0)
        for i in range(con_path.shape[0]):
            con_sin = np.delete(con_path[i], heal_link)
            sin_con_rm = np.append(sin_con_rm, con_sin)
        if sin_con_rm.shape[0] != 0:
            sin_con_rm = sin_con_rm.reshape(con_path.shape[0], -1).astype(int)
        con_rm.append(sin_con_rm)

    return con_rm, heal_link_total
