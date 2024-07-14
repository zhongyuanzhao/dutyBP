import os
import argparse
import sys
import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from backpressure import *

# input arguments calling from command line
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="../data_poisson_10", type=str, help="test data directory.")
parser.add_argument("--gtype", default="poisson", type=str, help="graph type.")
parser.add_argument("--size", default=10, type=int, help="size of dataset")
parser.add_argument("--seed", default=500, type=int, help="initial seed")
parser.add_argument("--m", default=8, type=int, help="node density")
args = parser.parse_args()

data_path = args.datapath
gtype = args.gtype
size = args.size
seed0 = args.seed
m = args.m 
# Create fig folder if not exist
if not os.path.isdir(data_path):
    os.mkdir(data_path)


def poisson_graph(size, nb=4, radius=1.0, seed=None):
    """
    Create a Poisson point process 2D graph
    """
    N = int(size)
    density = float(nb)/np.pi
    area = float(N) / density
    side = np.sqrt(area)
    if seed is not None:
        np.random.seed(int(seed))
    xys = np.random.uniform(0, side, (N, 2))
    d_mtx = distance_matrix(xys, xys)
    adj_mtx = np.zeros([N, N], dtype=int)
    adj_mtx[d_mtx <= radius] = 1
    np.fill_diagonal(adj_mtx, 0)
    graph = nx.from_numpy_matrix(adj_mtx)
    return graph, xys


graph_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
# graph_sizes = [20, 30, 40, 50, 60]

for num_nodes in graph_sizes:
    sidx = 0
    cnt = 0
    while cnt < size:
        seed = sidx + seed0
        # seed = id + 200
        # m = 8 # all the tests in first draft of manuscript
        # m = 12 # to address reviewer's comment on varying node densities
        sidx += 1
        # num_nodes = np.random.choice([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
        # num_nodes = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
        # bp_env = Backpressure(num_nodes, 100, seed=seed, m=m, pos='new', gtype=gtype)
        graph, pos_c = poisson_graph(num_nodes, nb=m, seed=seed)
        if not nx.is_connected(graph):
            continue
        cnt += 1
        adj = nx.adjacency_matrix(graph)
        flows_perc = np.random.randint(15, 30)
        num_flows = round(flows_perc/100 * num_nodes)
        nodes = graph.nodes()
        num_links = len(graph.edges())
        num_arr = np.random.permutation(nodes)
        arrival_rates = np.random.uniform(0.2, 1.0, (num_flows,))
        # link_rates = np.random.randint(12, (num_flows,))
        link_rates = np.random.uniform(10, 42, size=(num_links,))

        flows = []
        for fidx in range(num_flows):
            src = num_arr[2*fidx]
            dst = num_arr[2*fidx+1]
            flow = {'src': src, 'dst': dst, 'rate': arrival_rates[fidx]}
            flows.append(flow)

        # bp_env.flows_init()
        filename = "poisson_graph_seed{}_m{}_n{}_f{}.mat".format(seed, m, num_nodes, num_flows)
        filepath = os.path.join(data_path, filename)
        sio.savemat(filepath,
                    {"network": {"num_nodes": num_nodes, "seed": seed, "m": m},
                     "adj": adj.astype(np.float),
                     "link_rate": link_rates,
                     "flows": flows,
                     "pos_c": pos_c
                    })






