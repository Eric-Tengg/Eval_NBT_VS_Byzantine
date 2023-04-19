import numpy as np
from treeDrawn import *
from draw import *
from tqdm import tqdm


def scfs_algorithm(A_rm: np.ndarray, path_obs: np.ndarray):
    result = np.zeros(path_obs.shape[0], dtype=object)
    link_states_estimated_sam = np.zeros(A_rm.shape[1], dtype=int)
    for i in range(path_obs.shape[0]):  # 轮次为时间状态数
        y = path_obs[i]
        link_health = []
        link_states_estimated = link_states_estimated_sam.copy()
        for j in range(y.shape[0]):
            if int(y[j]) == 0:  # 该条路径正常
                for k in range(A_rm.shape[1]):
                    if int(A_rm[j][k]) == 1:
                        if k not in link_health:
                            link_health.append(k)
        for j in range(y.shape[0]):
            if int(y[j]) == 1:  # 该条路径拥塞
                for k in range(A_rm.shape[1]):
                    if int(A_rm[j][k]) == 1:
                        if link_states_estimated[k] == 1:
                            break
                        if k not in link_health:
                            link_states_estimated[k] = 1
                            break
        result[i] = np.where(link_states_estimated == 1)[0] + 1
    # result = result.reshape(path_obs.shape[0], -1).transpose()
    # result = result.astype(np.int64)
    return result


# 依据 Fig.2 给的算法伪代码逻辑，采用递归的方式来实现
def alg_SCFS(A_rm: np.ndarray, paths_obs: np.ndarray, queue=None):
    '''
        - 在树型拓扑中，对于每条链路 (s, d)，其可借由它的末端节点 link_d 来唯一指代，而不引起歧义
        - 正常/拥塞状态：0/1，注意与论文中相反
        - 链路/路径变量: X/Y，注意与论文中相反

        * Python 内部函数以及变量深拷贝参见
            = https://blog.51cto.com/u_14246112/3157550
            = https://blog.csdn.net/DeniuHe/article/details/77370112
    '''
    from copy import deepcopy

    W_s = np.zeros(paths_obs.shape[0], dtype=object)
    for idx, path_obs in enumerate(paths_obs):
        def recurse(k):  # 内部函数：在树型拓扑中递归地诊断链路 link_k
            nonlocal X, Y, W, R
            if k in R:
                X[k - 1] = Y[np.where(R == k)[0][0]]
            else:
                childNodeSet = d(k)
                for j in childNodeSet:
                    recurse(j)
                if k != 0:
                    X[k - 1] = min([X[j - 1] for j in childNodeSet])
                for j in childNodeSet:
                    if X[j - 1] == 1 and X[k - 1] == 0:
                        W.append(j)
                    if all(X):
                        W.append(1)

        def d(k):  # 内部函数：在树型拓扑中获取节点 k 的 child node 集合
            nonlocal linkSet_in_eachPath
            if k == 0:
                childNode_set = np.array([1])
            else:
                childNode_set = np.array(
                    [linkSet_in_eachPath[i][np.where(linkSet_in_eachPath[i] == k)[0][0] + 1] for i in
                     range(linkSet_in_eachPath.shape[0]) if
                     np.where(linkSet_in_eachPath[i] == k)[0].shape[0] != 0])
                childNode_set = np.unique(childNode_set)
            return childNode_set

        _, num_links = A_rm.shape  # 获取链路数量

        Y = deepcopy(path_obs)  # 路径状态观测值（已给定）；可以不进行 deepcopy 操作，因为整个程序不涉及修改 Y 的值；仅做提示作用
        X = np.zeros(num_links, dtype=int)  # 链路状态值（待估计）
        R = np.array([i + 1 for i in range(A_rm.shape[1]) if np.where(A_rm[:, i] == 1)[0].shape[0] == 1],
                     dtype=int)  # 叶节点
        linkSet_in_eachPath = np.array([np.where(A_rm[i] == 1)[0] + 1 for i in range(A_rm.shape[0])],
                                       dtype=object)  # 每条路径上的链路集

        W = []

        recurse(0)
        W_s[idx] = np.unique(np.array(W))
    if queue is None:
        return W_s
    else:
        queue.put(W_s)


if __name__ == '__main__':
    # 示例
    A_rm = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]])
    # A_rm = get_tree_from_gml('AsnetAm.gml')
    # print(A_rm)
    # path_obs = np.array((np.random.rand(A_rm.shape[0]) < 0.5) * 1).reshape(1, -1)
    # print(path_obs)
    # path_obs = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 1]])
    # path_obs = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    path_obs = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                         [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]])
    link_stat_inferred = scfs_algorithm(A_rm, path_obs)
    print(f'SCFS算法诊断:\n{link_stat_inferred}')
    link_stat_inferred = alg_SCFS(A_rm, path_obs)
    print(f'SCFS算法诊断:\n{link_stat_inferred}')
    # draw_topo(A_rm, np.zeros(A_rm.shape[1], dtype=int), np.zeros(A_rm.shape[1], dtype=int),
    #           path_obs[0], )
