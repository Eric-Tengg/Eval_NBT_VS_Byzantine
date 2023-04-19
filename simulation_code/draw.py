from scfs import *
import networkx as nx
import matplotlib.pyplot as plt


def draw_topo(A_rm: np.ndarray,
              links_state: np.ndarray = None, links_state_inferred: np.ndarray = None,
              paths_state: np.ndarray = None, paths_attacked=None,
              performance=None, highlight: tuple = None):
    # 创建图像
    # 设置图像参数
    if paths_attacked is not None:
        paths_truth = paths_state.copy()
        paths_state = paths_attacked.copy()
        links_state_inferred = scfs_algorithm(A_rm, paths_state.reshape(1, -1)).transpose()[0]
    if links_state is None:
        links_state = np.zeros(A_rm.shape[1], dtype=int)
        links_state_inferred = np.zeros(A_rm.shape[1], dtype=int)
        paths_state = np.zeros(A_rm.shape[0], dtype=int)
        paths_attacked = np.zeros(A_rm.shape[0], dtype=int)
        paths_truth = paths_state.copy()
    tree_vector = rm_tv(A_rm)
    vertices = tree_vector.shape[0] + 1
    # print(A_rm)
    idx = list(range(vertices))  # 转换成列表

    old_edges = rm_edges(A_rm)
    # 对edges进行排序
    rank = np.argsort(old_edges[:, 1])
    edges = np.zeros(rank.shape[0], dtype=np.ndarray)
    for i, id in enumerate(rank):
        edges[i] = old_edges[id]
    # print(edges)
    leaf_nodes = rm_leaf_node(A_rm) + 1
    # print(leaf_nodes)

    g = nx.DiGraph()
    g.add_nodes_from(idx)
    g.add_edges_from(edges)

    pos = nx.nx_pydot.pydot_layout(g, prog='dot')

    # 可以适当地封装
    # 设置节点颜色字典，将坏路径的目的节点设为"b"
    vcolor_dict = {"n": "#808080", "g": "#228B22", "b": "#CD5C5C"}

    bad_paths = []
    for i in range(paths_state.shape[0]):
        if paths_state[i] == 1:
            bad_paths.append(i + 1)

    if paths_attacked is not None:
        byzantine_path = []
        for k in range(paths_attacked.shape[0]):
            if paths_truth[k] != paths_attacked[k]:
                byzantine_path.append(k)
        byzantineEndNodes = [leaf_nodes[i] for i in byzantine_path]

    badPathEndNodes = [leaf_nodes[i - 1] for i in bad_paths]
    # print("坏路径节点:\n", badPathEndNodes)

    v_num = A_rm.shape[1] + 1
    vertexs_color = np.zeros(v_num, dtype=str)
    for i in range(v_num):
        if i in badPathEndNodes:
            vertexs_color[i] = 'b'
        elif i in leaf_nodes:
            vertexs_color[i] = 'g'
        else:
            vertexs_color[i] = 'n'
    # print(vertexs_color)

    # 将链路分为四类，一是链路状态是否判断正确，二是实际链路状态是否正常
    correct_normal = []
    correct_congested = []
    wrong_normal = []
    wrong_congested = []

    for i in range(links_state.shape[0]):
        if links_state[i] == 1:
            if links_state_inferred[i] == 1:
                correct_congested.append(edges[i])
            else:
                wrong_congested.append(edges[i])
        else:
            if links_state_inferred[i] == 1:
                wrong_normal.append(edges[i])
            else:
                correct_normal.append(edges[i])
    # 创建图像可视化字典
    nodes_style = {"pos": pos,
                   "node_size": 100,
                   "node_shape": 'o',
                   "node_color": [vcolor_dict[obs] for obs in vertexs_color]}
    nx.draw_networkx_nodes(g, **nodes_style)
    if paths_attacked is not None:
        g.remove_nodes_from(byzantineEndNodes)
        vertexs_color_byzantine = np.zeros(len(byzantineEndNodes), dtype=str)
        for i in range(len(byzantineEndNodes)):
            if byzantineEndNodes[i] in badPathEndNodes:
                vertexs_color_byzantine[i] = 'b'
            else:
                vertexs_color_byzantine[i] = 'g'
        nodes_style_byzantine = {"pos": pos,
                                 "nodelist": byzantineEndNodes,
                                 "node_size": 100,
                                 "node_shape": 's',
                                 "node_color": [vcolor_dict[obs] for obs in vertexs_color_byzantine]}
        nx.draw_networkx_nodes(g, **nodes_style_byzantine)
    correct_normal_estyle = {"pos": pos,
                             "edgelist": correct_normal,
                             "edge_color": "#90EE90",
                             "arrowsize": 10,
                             'width': 1.5,
                             "style": 'solid'}
    correct_congested_estyle = {"pos": pos,
                                "edgelist": correct_congested,
                                "edge_color": "#CD5C5C",
                                "arrowsize": 10,
                                'width': 1.5,
                                "style": 'solid'}

    wrong_normal_estyle = {"pos": pos,
                           "edgelist": wrong_normal,
                           "edge_color": "#90EE90",
                           "arrowsize": 10,
                           'width': 1.5,
                           "style": 'dashed'}
    wrong_congested_estyle = {"pos": pos,
                              "edgelist": wrong_congested,
                              "edge_color": "#CD5C5C",
                              "arrowsize": 10,
                              'width': 1.5,
                              "style": 'dashed'}

    nx.draw_networkx_edges(g, **correct_normal_estyle)
    nx.draw_networkx_edges(g, **correct_congested_estyle)
    nx.draw_networkx_edges(g, **wrong_normal_estyle)
    nx.draw_networkx_edges(g, **wrong_congested_estyle)
    nx.draw_networkx_labels(
        g, pos, labels={i: i for i in idx}, font_size=7, font_color='#FFFFFF')  # 画标签

    plt.title(performance)
    if performance is None:
        plt.text(1.0, -0.5, performance, size=12,
                 bbox=dict(boxstyle="round", fc="gray", ec="1.0", alpha=0.2))
    if highlight:
        highlight_edges = []
        if highlight[0] == 0:
            highlight_edges.append((0, 1))
            highlight = (1, highlight[1])
        for i in range(A_rm.shape[0]):
            if A_rm[i][highlight[0] - 1] == 1 and A_rm[i][highlight[1] - 1] == 1:
                highlight_path = A_rm[i]
                edge_nodes = np.where(highlight_path == 1)[0]
                endIndex = edge_nodes.tolist().index(highlight[1] - 1)
                highlight_edges = [(edge_nodes[i] + 1, edge_nodes[i + 1] + 1) for i in range(endIndex)]
                break
        draw_route = {
            "pos": pos,
            "edgelist": highlight_edges,
            "edge_color": "yellow",
            "arrowsize": 10,
            'width': 1.5,
            'alpha': 0.6,
            "style": 'solid'
        }
        nx.draw_networkx_edges(g, **draw_route)
    plt.axis('off')  # 去除图像边框
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 使图像占满输出框
    plt.show()
