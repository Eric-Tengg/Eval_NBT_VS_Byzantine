# 抽取节点最多树拓扑
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns


def gen_tree_f_mul(source, edges, tree_edges):
    # 输出以该节点为根节点的树拓扑边集
    if source in edges[:, 0]:
        for idx in np.where(edges[:, 0] == source)[0]:
            # 如果该节点的子节点已经在边集的子节点中，则跳过
            if tree_edges.shape[0] == 0:
                tree_edges = np.append(tree_edges, edges[idx]).reshape(-1, 2)
            elif edges[idx, 1] not in tree_edges[:, 1]:
                tree_edges = np.append(tree_edges, edges[idx]).reshape(-1, 2)
            tree_edges = gen_tree_f_mul(edges[idx, 1], edges, tree_edges)
        # print(tree_edges)
    return tree_edges


def del_out1(tree_edges, ):
    # 删去树拓扑中出度为1的节点
    old_nodes = np.unique(np.append(tree_edges[:, 0], tree_edges[:, 1]))
    for i in range(1, old_nodes.shape[0]):
        node = old_nodes[i]
        loc = np.where(tree_edges[:, 0] == node)[0]
        if loc.shape[0] == 1:
            tree_edges = np.delete(tree_edges, loc[0], 0)
    # 得到所有节点集
    old_nodes = np.unique(np.append(tree_edges[:, 0], tree_edges[:, 1]))
    for i in range(1, old_nodes.shape[0]):
        node = old_nodes[i]
        if node not in tree_edges[:, 1]:
            loc = np.where(tree_edges[:, 0] == node)[0]
            tree_edges = np.delete(tree_edges, loc, 0)

    return tree_edges


def max_tree_drawn(edges):
    max_tree_edges = np.zeros(0, dtype=int)
    # start_time = time.time()
    for root in range(max(edges[:, 1]) + 1):
        # if time.time() - start_time > 2:
        #     break
        tree_edges = np.zeros(0, dtype=int)
        # 得到以root为根节点的树
        tree_edges = gen_tree_f_mul(root, edges, tree_edges)
        tree_edges = np.unique(np.array(tree_edges), axis=0)
        # 剪枝 删去出度为1的节点
        if tree_edges.shape[0] != 0:
            tree_edges = del_out1(tree_edges)
        # # 筛选最大树（链路数优先）
        # if tree_edges.shape[0] > max_tree_edges.shape[0]:
        #     max_tree_edges = tree_edges.astype(int)
        # 筛选最大树（树深度优先）
        if max_tree_edges.shape[0] == 0:
            max_tree_edges = tree_edges
        elif tree_edges.shape[0] > 1 and max_tree_edges.shape[0] > 1:
            path = tv_path(edges_tv(tree_edges))  # tree_edges
            path_max = tv_path(edges_tv(max_tree_edges))  # max_tree_edges
            length, length_max = 0, 0
            for i in path:
                length = len(i) if len(i) > length else length
            for i in path_max:
                length_max = len(i) if len(i) > length_max else length_max
            max_tree_edges = tree_edges if length > length_max else max_tree_edges

    return max_tree_edges


def edges_tv(tree_edges):
    # # 将边集转换为树向量
    # # # 得到所有节点集
    # print("边集:\n", tree_edges)
    old_nodes = np.unique(np.append(tree_edges[:, 0], tree_edges[:, 1]))
    # print(old_nodes)
    old_tree = tree_edges.copy()
    # # # 更改序号
    if tree_edges.shape[0] > 1:
        m = 1 if tree_edges[0, 0] == tree_edges[1, 0] else 0  # 判断是否额外增加根节点0
        for i in range(old_nodes.shape[0]):
            for j in range(2):
                for k in np.where(old_tree[:, j] == old_nodes[i])[0]:
                    tree_edges[k, j] = i + m
        if m == 1:
            tree_edges = np.append([0, 1], tree_edges).reshape(-1, 2)
        # print("树链路:\n", tree_edges)
        link_num = max(tree_edges[:, 1])
        # print("链路数:\n", link_num)
        tree_vector = np.zeros(link_num, dtype=int)
        for i in range(link_num):
            # print(np.where(tree_edges[:, 1] == i + 1)[0][0])
            idx = np.where(tree_edges[:, 1] == i + 1)[0][0]
            # print(i + 1, idx, tree_vector)
            tree_vector[i] = tree_edges[idx, 0]
        # print(tree_vector)
    else:
        tree_vector = np.array([0])
    return tree_vector


def tv_path(tree_vector):
    # 由树向量遍历得到各路径上链路集
    # # # 通过树向量得到叶节点
    leafNodes = []
    for i in range(1, tree_vector.shape[0] + 1):

        if i not in tree_vector:
            leafNodes.append(i)
    # print("leaf:\t", leafNodes)

    # # # 得到各条路径
    mul_path = []
    for i in range(len(leafNodes)):
        sin_path = [leafNodes[i]]
        fNode = tree_vector[leafNodes[i] - 1]
        while fNode != 0:
            sin_path.append(fNode)
            fNode = tree_vector[fNode - 1]
        mul_path.append(sin_path)
    # print("遍历各路径上链路:\n", mul_path)
    return mul_path


def rm_edges(routine_matrix):
    # 得到边集
    edges = []
    for i in range(routine_matrix.shape[0]):
        path = routine_matrix[i]
        path_edge = np.where(path == 1)[0] + 1
        for j in range(path_edge.shape[0] - 1):
            edges.append(np.array([path_edge[j], path_edge[j + 1]]))
    edges.append(np.array([0, 1]))
    edges = np.unique(np.array(edges), axis=0)
    return edges


def rm_tv(routine_matrix):
    # 得到边集
    edges = rm_edges(routine_matrix)
    # 得到树向量
    tree_vector = np.zeros(edges.shape[0], dtype=int)
    for i in range(edges.shape[0]):
        idx = np.where(edges[:, 1] == i + 1)[0][0]
        tree_vector[i] = edges[idx, 0]
    return tree_vector


def tv_rm(tree_vector):
    # # 将树向量转化为路由矩阵
    mul_path = tv_path(tree_vector)
    leafNodes = []
    for i in range(1, tree_vector.shape[0] + 1):
        if i not in tree_vector:
            leafNodes.append(i)
    A_rm = np.zeros(tree_vector.shape[0] * len(leafNodes), dtype=int).reshape(len(leafNodes), -1)
    # # # 得到路由矩阵
    for i in range(len(mul_path)):
        sin_path = mul_path[i]
        for j in sin_path:
            A_rm[i][j - 1] = 1

    # print("路由矩阵:\n", A_rm)
    return A_rm


def tree_drawn(edges: np.ndarray):
    # 得到最大树的边集
    # edges = pd.DataFrame(pd.read_csv(topo_path), columns=['Source', 'Target']).to_numpy(dtype=int)
    # print(edges)

    # 筛选得到最大树拓扑
    tree_edges = max_tree_drawn(edges)
    # print(tree_edges)

    # 将边集转换为路由矩阵
    # # 将边集转换为树向量
    tree_vector = edges_tv(tree_edges)
    # print(f'树向量(链路数:{tree_vector.shape[0]}):\n{tree_vector}')

    # # 将树向量转化为路由矩阵
    A_rm = tv_rm(tree_vector)
    # print(f'路由矩阵(路径数:{A_rm.shape[0]}, 链路数:{A_rm.shape[1]}):\n{A_rm}')

    return A_rm


def getWords(filepath):
    file = open(filepath)
    wordOne = []
    while file:
        line = file.readline()
        word = line.split('/')
        wordOne.extend(word)
        if not line:  # 若读取结束了
            break
    wordtwo = []
    for i in wordOne:
        wordtwo.extend(i.split())
    return wordtwo


def getWordNum(words):
    dictWord = {}
    for i in words:
        if i not in dictWord:
            dictWord[i] = 0
        dictWord[i] += 1
    return dictWord


def gml_edges(gml):
    # 从gml文件得到边集
    g = Graph.Read_GML(gml)
    edges = np.array(g.get_edgelist())
    return edges


def rm_leaf_node(routine_matrix):
    leaf = np.zeros(routine_matrix.shape[0], dtype=int)
    for i in range(routine_matrix.shape[0]):
        path = routine_matrix[i]
        leaf[i] = np.where(path == 1)[0][-1]
    return leaf


def abstract_tree(root_folder_path=os.path.join(os.path.dirname(__file__), 'datasets'), if_ret_maxtree=False):
    os.makedirs(os.path.join(root_folder_path, 'topo_abstract'), exist_ok=True)
    filelist = os.listdir(os.path.join(root_folder_path, 'topology_zoo'))  # The path stored the GML files.
    gml_file = [item for item in filelist if 'gml' in item]  # The name list stored GML file names
    size = np.zeros(len(gml_file))
    tree = []
    # 得到所有树
    for i in range(len(gml_file)):
        gml_path = os.path.join('dataset', gml_file[i])
        print('已获取拓扑:', gml_file[i], end=" ")
        edges = gml_edges(gml_path)
        topo = tree_drawn(edges)
        if any(np.array_equal(topo, arr) for arr in tree):
            tree.append(topo)
            size[i] = np.nan
        else:
            tree.append(topo)
            size[i] = tree[i].shape[1]
            savePath = os.path.join(root_folder_path, 'topo_abstract', f"{gml_file[i].split('.')[0]}.csv")
            pd.DataFrame(topo).to_csv(savePath, index=False, header=False)
        print('链路数为{}'.format(size[i]))
    # 画出树深度分布
    sns.set_style('whitegrid')
    sns.countplot(x=size.astype(int))
    # 获取每个柱形图的高度
    heights = [int(p.get_height()) for p in plt.gca().patches]
    # 在每个柱形图上添加具体数值的注释
    for i, height in enumerate(heights):
        plt.gca().annotate(str(height), xy=(i, height), ha='center', va='bottom')

    plt.xlabel('Depth')
    plt.ylabel('Count')
    plt.title('{} trees'.format(sum(heights)))
    plt.show()
    if if_ret_maxtree:
        # 得到最大树深度
        max_depth = max(size)
        # print("最大树深度:\n", max_depth)
        # 得到最大深度树
        idx = np.where(size == max_depth)[0][0]
        max_tree = tree[idx]
        print(
            f'最大深度树(拓扑名:{gml_file[idx]}, 路径数:{max_tree.shape[0]}, 链路数:{max_tree.shape[1]}):\n{max_tree}')
        return max_tree


def get_tree_from_gml(gml_name: str):
    gml_path = os.path.join('./dataset', gml_name)
    print('已获取拓扑:', gml_name)
    edges = gml_edges(gml_path)
    tree = tree_drawn(edges)
    return tree


def draw(routine_matrix):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(routine_matrix.shape[1] + 1)))
    G.add_edges_from(rm_edges(routine_matrix))
    plt.rcParams['figure.figsize'] = (5, 4)
    pos = nx.nx_pydot.pydot_layout(G, prog='dot')
    nx.draw(G, with_labels=True, pos=pos)
    plt.show()


if __name__ == '__main__':
    abstract_tree()  # abstract the trees from Topology-Zoo
    # tree = abstract_tree(True)  # Gain the max tree
    # print(tree)
    # gml_name = 'AsnetAm.gml'
    # A_rm = get_tree_from_gml(gml_name)
    # print(f'最大深度树(拓扑名:{gml_name}, 路径数:{A_rm.shape[0]}, 链路数:{A_rm.shape[1]}):\n{A_rm}')
    # tv = rm_tv(A_rm)
    # print(f'树向量(链路数:{tv.shape[0]}):\n{tv}')
    # draw(A_rm)
