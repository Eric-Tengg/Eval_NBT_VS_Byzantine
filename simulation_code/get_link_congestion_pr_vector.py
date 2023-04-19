import numpy as np
import pickle
from tqdm import tqdm
import os
import pandas as pd
import copy as cp


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def generate(num_links=16, filename='link_congestion_pro_vector.pickle'):
    scenario = {'场景-1: 不变方差': [((i - 0.1).round(2), i.round(2)) for i in np.linspace(0.1, 1.0, 10)],
                '场景-2: 可变方差': [(0.0, i.round(2)) for i in np.linspace(0.1, 1.0, 10)]}
    sample_times = 100  # 每个区间产生 100 次

    link_congestion_pro_vector = {}
    for scn in scenario:
        for congestion_interval in scenario[scn]:
            for t in range(sample_times):
                link_congestion_pro_vector[scn, congestion_interval, t] = np.random.uniform(congestion_interval[0],
                                                                                            congestion_interval[1],
                                                                                            num_links)
    os.makedirs('network_data', exist_ok=True)
    with open(os.path.join(get_root_path(), 'data', 'network_data', filename), 'wb') as handle:
        pickle.dump(link_congestion_pro_vector, handle)

    return link_congestion_pro_vector


def generate_path_pro(filename='path_congestion_pro_vector.pickle'):
    # based on link_congestion_pro_vector.pickle
    routing_matrix = pd.read_csv(os.path.join(get_root_path(), '10_16_routing_matrix.csv'), header=None).to_numpy()
    link_congestion_pro_vector = load()
    scenario = {'场景-1: 不变方差': [((i - 0.1).round(2), i.round(2)) for i in np.linspace(0.1, 1.0, 10)],
                '场景-2: 可变方差': [(0.0, i.round(2)) for i in np.linspace(0.1, 1.0, 10)]}
    sample_times = 100  # 每个区间产生 100 次

    path_congestion_pro_vector = {}
    for scn in scenario:
        for congestion_interval in scenario[scn]:
            for t in tqdm(range(sample_times), desc=f'{scn}, {congestion_interval}'):
                pro = link_congestion_pro_vector[scn, congestion_interval, t]
                pathstatus_pro_set = np.zeros(1024, dtype=float)
                linkstatus_setNum = pow(2, 16)
                for i in range(linkstatus_setNum):
                    bin_str = bin(i)[2:]
                    # Pad the binary string with leading zeros to a length of 8
                    padded_bin_str = bin_str.zfill(16)
                    # Convert the padded binary string to a binary numpy array
                    linkstatus = np.array(list(padded_bin_str), dtype=np.uint8)
                    status_pro = np.product([pro[j] if linkstatus[j] == 1 else 1 - pro[j] for j in range(16)])
                    pathstatus = [(np.dot(linkstatus, routing_matrix[j]) != 0) * 1 for j in range(10)]
                    dec = int(''.join(map(str, pathstatus)), 2)  # 出现的路径情况
                    pathstatus_pro_set[dec] += status_pro

                path_congestion_pro_vector[scn, congestion_interval, t] = cp.deepcopy(pathstatus_pro_set)
    with open(os.path.join(get_root_path(), 'data', 'network_data', filename), 'wb') as handle:
        pickle.dump(path_congestion_pro_vector, handle)

    return path_congestion_pro_vector


def load(num_links=16, filename='link_congestion_pro_vector_16.pickle'):
    try:
        with open(os.path.join(get_root_path(), 'data', 'network_data', filename), 'rb') as handle:
            link_congestion_pro_vector = pickle.load(handle)

        return link_congestion_pro_vector

    except:  # 如果在当前文件夹里没有产生场景配置文件，则重新生成
        return generate(num_links, filename)


if __name__ == '__main__':
    for i in np.arange(3, 25, 2):
        generate(num_links=i, filename='link_congestion_pro_vector_{}.pickle'.format(i))

    # link_congestion_pro_vector = load(num_links=12, filename='link_congestion_pro_vector_12.pickle')
    # pass
    # generate_pickle(filename=)

