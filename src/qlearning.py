# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import numpy as np


def QLearning(bp, num_episodes=5, alpha=0.1, gamma=0.9):
    N_links = bp.num_links
    N_nodes = bp.num_nodes
    Q_mtx = np.zeros((N_links * 2, N_nodes), dtype=float) # Q_ij does not equal to Q_ji
    BiasQ_mtx = np.inf * np.ones_like(bp.queue_matrix, dtype=float)
    for c in bp.dst_nodes:
        for ith_episode in range(num_episodes):
            for l in range(bp.num_links):
                link = bp.link_list[l]
                i, j = link
                Q_min_jk = Minimum_Qjk(Q_mtx[0:N_links, c], j, bp)
                # store link (i, j) at l, because (i, j) is in link_list
                Q_mtx[l, c] = (1 - alpha) * Q_mtx[l, c] + alpha * (bp.queue_matrix[j, c] + gamma * Q_min_jk)
                Q_min_ik = Minimum_Qjk(Q_mtx[N_links:2 * N_links, c], i, bp)
                # store link (j, i) at l + N_links
                Q_mtx[l+N_links, c] = (1 - alpha) * Q_mtx[l+N_links, c] + alpha * (bp.queue_matrix[i, c] + gamma * Q_min_ik)

        for i in range(N_nodes):
            links = []
            for j in bp.graph_c.neighbors(i):
                link0 = (i, j)
                link1 = (j, i)
                if link0 in bp.link_list:
                    l = bp.link_list.index(link0)
                elif link1 in bp.link_list:
                    l = bp.link_list.index(link1) + N_links
                else:
                    continue
                links.append(l)
            BiasQ_mtx[i, c] = np.amin(Q_mtx[links, c])
    return BiasQ_mtx, Q_mtx


def Minimum_Qjk(Q_c, j, bp):
    links = []
    for k in bp.graph_c.neighbors(j):
        link0 = (j, k)
        link1 = (k, j)
        if link0 in bp.link_list:
            l = bp.link_list.index(link0)
        elif link1 in bp.link_list:
            l = bp.link_list.index(link1)
        else:
            continue
        links.append(l)
    Q_min = np.amin(Q_c[links])
    return Q_min
