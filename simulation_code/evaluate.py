import numpy as np
from sklearn.metrics import confusion_matrix


def fbeta_score(links_state, links_state_inferred, beta=1):
    """
    对 精确率(precision) 和 召回率(recall) 设置 权重(beta) 进行算法的性能评估

    Parameters
    ----------
    links_state : array_like
        二维数组, 元素type = object，Shape = (时刻数, ), 每个 array 记录真实拥塞链路序号
    links_state_inferred : array_like
        二维数组, 元素type = object, Shape = (时刻数, ), 每个 array 记录预测拥塞链路序号
    beta : float
        控制 精确率 和 召回率 的相对权重,
            当 beta = 1 时, 精确率 和 召回率 的权重是相等的, 数值等于 F1;
            当 beta < 1 时, 更注重 精确率;
            当 beta > 1 时, 更注重 召回率;

    Returns
    -------
    F_beta : array_like
        一维数组, Shape = (时刻数), 记录各时刻的性能评估

    Notes
    -----
    precision = \frac{TP}{( TP + FP )}, recall = \frac{TP}{( TP + FN )}
    F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}
                                        {(\beta^2 \cdot \text{precision}) + \text{recall}}
        when \beta -> 0, only precision, \beta -> inf, only recall

    Examples
    --------
    links_state = np.array([np.array([3, 4]), np.array([7, 9]), np.array([1, 2, 7])], dtype=object)
    links_state_inferred = np.array([np.array([3]), np.array([7, 8, 9]), np.array([1, 2, 7])], dtype=object)
    beta = 0.5
    > fbeta_score(links_state, links_state_inferred, beta) = array[0.83333333 0.71428571 1.        ]

    """
    F_beta = np.zeros(links_state.shape[0], dtype=float)
    for i in range(links_state.shape[0]):
        link_stat = links_state[i].astype(int)
        link_stat_inf = links_state_inferred[i].astype(int)
        if np.array_equal(link_stat, link_stat_inf):
            precision = 1
            recall = 1
        elif link_stat.shape[0] == 0 or link_stat_inf.shape[0] == 0:
            precision = 0
            recall = 0
        else:
            TP = len(np.intersect1d(link_stat, link_stat_inf))
            precision = TP / link_stat_inf.shape[0]
            recall = TP / link_stat.shape[0]
        if precision + recall == 0:
            F_beta[i] = 0
        else:
            beta_2 = pow(beta, 2)
            F_beta[i] = (1 + beta_2) * (precision * recall / (beta_2 * precision + recall))
    return F_beta


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


def detection(links_state, links_state_inferred):
    DRs = np.zeros(links_state.shape[0], dtype=float)
    FPRs = np.zeros(links_state.shape[0], dtype=float)
    for i in range(links_state.shape[0]):
        if np.array_equal(links_state[i], links_state_inferred[i]):
            DR = 1
            FPR = 0
        else:
            TP = len(np.intersect1d(links_state[i], links_state_inferred[i]))
            FN = np.setdiff1d(links_state[i], links_state_inferred[i]).shape[0]
            FP = np.setdiff1d(links_state_inferred[i], links_state[i]).shape[0]
            N = 16 - TP - FN
            if N == 0:
                DR = TP / (TP + FN)
                FPR = 0
            elif TP != 0:
                DR = TP / (TP + FN)
                FPR = FP / N
            else:
                DR = 0
                FPR = FP / N
        DRs[i] = DR
        FPRs[i] = FPR
    return DRs, FPRs


def detect(link_inferred, link, num, n):
    # DR是检出率，fpr是假阳率
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(num):
        diagnose_link = np.zeros(n, dtype=int)
        real_link = np.zeros(n, dtype=int)
        diagnose_link[link_inferred[i] - 1] = 1
        real_link[link[i] - 1] = 1
        for j in range(n):
            if diagnose_link[j] == 1 and real_link[j] == 1:
                TP = TP + 1
            elif diagnose_link[j] == 1 and real_link[j] == 0:
                FP = FP + 1
            elif diagnose_link[j] == 0 and real_link[j] == 1:
                FN = FN + 1
            else:
                TN = TN + 1
            # print(diagnose_link[i][j], real_link[i][j], "TP", TP, "FP", FP, "FN", FN, "TN", TN)
    # 修改DR和FPR
    DR = TP / (TP + FN)
    FPR = FP / (TP + FP)
    # print(DR,FPR)
    return DR, FPR


if __name__ == '__main__':
    # links_state = np.array([np.array([1, 2, 3]), np.array([1, 3])], dtype=object)
    # links_state_inferred = np.array([np.array([1, 2, 3]), np.array([2, 3])], dtype=object)
    # DRs, FPRs = detection(links_state, links_state_inferred)
    # print(DRs, FPRs)
    # test fbeta_score
    # links_state = np.array([np.array([3, 4]), np.array([7, 9]), np.array([1, 2, 7])], dtype=object)
    # links_state_inferred = np.array([np.array([3]), np.array([7, 8, 9]), np.array([1, 2, 7])], dtype=object)
    beta = 0.5
    # print(fbeta_score(links_state, links_state_inferred, beta))
