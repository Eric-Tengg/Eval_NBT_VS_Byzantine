import json
from joblib import Parallel, delayed
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def get_root_path():
    root_path = os.path.abspath('./')
    while not os.path.exists(os.path.join(root_path, 'README.md')):
        root_path = os.path.abspath(os.path.join(root_path, '..'))
    return root_path


def gen_true_perf(alg, interval, i, scn, topoName):
    filePath = os.path.join(get_root_path(), 'data', topoName, scn, f"{interval}-{i + 1}-1000",
                            f"{interval}-{i + 1}-#1000-linkstatus-{alg}-high-eval.csv")
    perf_set = ['DR', 'FPR', 'F1']
    for perf in perf_set:
        true_perf = pd.read_csv(filePath, index_col=0).loc[perf].mean()
        save_folderPath = os.path.join(get_root_path(), 'analysis', 'statistical_analysis',
                                       topoName, scn, 'true', perf, alg)
        os.makedirs(save_folderPath, exist_ok=True)
        savePath = os.path.join(save_folderPath, f"{interval}-{i + 1}.json")
        with open(savePath, 'w') as file:
            json.dump(true_perf, file)


def gen_att_perf(alg, interval, i, path, freq, scn, topoName):
    perf_set = ['DR', 'FPR', 'F1']
    for perf in perf_set:
        att_perf = 0
        for j in range(100):
            filePath = os.path.join(get_root_path(), 'data', topoName, scn, f"{interval}-{i + 1}-1000", "attack",
                                    "freq-topic", "path", f"path-{path}",
                                    f"{interval}-{i + 1}-#1000-linkstatus-attacked-{freq}-{j + 1}-{alg}-high-eval.csv")
            if freq != 1.0:
                att_perf += pd.read_csv(filePath, index_col=0).loc[perf].mean() / 100
            else:
                att_perf = pd.read_csv(filePath, index_col=0).loc[perf].mean()
                break

        save_folderPath = os.path.join(get_root_path(), 'analysis', 'statistical_analysis', topoName,
                                       scn, 'attack', perf, alg, f"{interval}-{i + 1}", f"path{path}")
        os.makedirs(save_folderPath, exist_ok=True)
        savePath = os.path.join(save_folderPath, f"freq{freq}.json")
        with open(savePath, 'w') as file:
            json.dump(att_perf, file)


def gen_perf(scn, topoName):
    if scn == 'even':
        intervals = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    else:
        intervals = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    freq_set = np.linspace(0.1, 1.0, 10).round(1)
    path_set = list(range(1, 11))
    alg_set = ['scfs', 'clink', 'map']

    for alg in alg_set:
        for interval in intervals:
            for i in tqdm(range(20), desc=f"{alg}-{interval}"):
                gen_true_perf(alg, interval, i, scn, topoName)
                Parallel(n_jobs=-1)(delayed(gen_att_perf)(alg, interval, i, path, freq, scn, topoName)
                                    for path in path_set for freq in freq_set)


def read_perf(scn, topoName):
    if scn == 'even':
        intervals = [(i, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    else:
        intervals = [(0, round(i + 0.1, 1)) for i in np.arange(0.0, 1.0, 0.1).round(1)]
    freq_set = np.linspace(0.1, 1.0, 10).round(1)
    path_set = list(range(1, 11))
    alg_set = ['scfs', 'clink', 'map']
    perf_set = ['DR', 'FPR', 'F1']
    data = {'true': {}, 'attack': {}}

    for perf in perf_set:
        data['true'][perf] = {}
        data['attack'][perf] = {}
        for alg in alg_set:
            data['true'][perf][alg.upper()] = {}
            data['attack'][perf][alg.upper()] = {}
            for interval in intervals:
                interval_key = str(interval)
                data['true'][perf][alg.upper()][interval_key] = {}
                data['attack'][perf][alg.upper()][interval_key] = {}
                for i in tqdm(range(20), desc=f"{perf}-{alg}-{interval}"):
                    sample_key = f"sample{i + 1}"
                    # Read
                    true_path = os.path.join(get_root_path(), 'analysis', 'statistical_analysis',
                                             topoName, scn, 'true', perf, alg)
                    with open(os.path.join(true_path, f"{interval}-{i + 1}.json"), 'r') as f:
                        data['true'][perf][alg.upper()][interval_key][sample_key] = json.load(f)

                    data['attack'][perf][alg.upper()][interval_key][sample_key] = {}
                    for path in path_set:
                        path_key = f"path{path}"
                        data['attack'][perf][alg.upper()][interval_key][sample_key][path_key] = {}
                        for freq in freq_set:
                            freq_key = f"freq{freq}"
                            # Read
                            att_path = os.path.join(get_root_path(), 'analysis', 'statistical_analysis', topoName,
                                                    scn, 'attack', perf, alg, f"{interval}-{i + 1}", f"path{path}")
                            with open(os.path.join(att_path, f"freq{freq}.json"), 'r') as f:
                                data['attack'][perf][alg.upper()][interval_key][sample_key][path_key][freq_key] = \
                                    json.load(f)

    return data


if __name__ == "__main__":
    for scn in ['even', 'uneven']:
        topo_path = os.path.join(get_root_path(), 'datasets', 'tree_topo')
        dirs = [fileName.split('.')[0] for fileName in os.listdir(topo_path)]
        for topoName in dirs:
            # 生成数据
            gen_perf(scn, topoName)
            # 读取数据
            data = read_perf(scn, topoName)
            # 存储
            folderPath = os.path.join(get_root_path(), 'analysis', 'statistical_analysis', topoName)
            with open(os.path.join(folderPath, f'{scn}.json'), 'w') as outfile:
                json.dump(data, outfile)
