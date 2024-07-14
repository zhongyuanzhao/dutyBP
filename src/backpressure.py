# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import queue
import sys
import os
import time
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scipy.io as sio
import sparse
np.set_printoptions(threshold=np.inf)
# Import utility functions
from util import *


class Backpressure:
    def __init__(self, num_nodes, T, seed=3, m=2, pos=None, cf_radius=0.0, gtype='ba', trace=False):
        self.num_nodes = int(num_nodes)
        self.T = int(T)
        self.seed = int(seed) # other format such as int64 won't work
        self.m = int(m)
        self.gtype = gtype.lower()
        self.trace = trace
        self.cf_radius = cf_radius
        self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, self.gtype)
        if self.gtype == 'ba':
            graph_c = nx.barabasi_albert_graph(self.num_nodes, self.m, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'grp':
            graph_c = nx.gaussian_random_partition_graph(self.num_nodes, 15, 3, 0.4, 0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'ws':
            graph_c = nx.connected_watts_strogatz_graph(self.num_nodes, k=6, p=0.2, seed=self.seed)  # Conectivity graph
        elif self.gtype == 'er':
            graph_c = nx.fast_gnp_random_graph(self.num_nodes, 15.0/float(self.num_nodes), seed=self.seed)  # Conectivity graph
        elif '.mat' in self.gtype:
            postfix = self.gtype.split('/')[-1]
            postfix = postfix.split('.')[0]
            self.case_name = 'seed_{}_nodes_{}_{}'.format(self.seed, self.num_nodes, postfix)
            try:
                mat_contents = sio.loadmat(self.gtype)
                adj = mat_contents['adj'].todense()
                pos = mat_contents['pos_c']
                graph_c = nx.from_numpy_array(adj)
            except:
                raise RuntimeError("Error creating object, check {}".format(self.gtype))
        else:
            raise ValueError("unsupported graph model for connectivity graph")
        self.connected = nx.is_connected(graph_c)
        self.graph_c = graph_c
        self.node_positions(pos)
        self.box = self.bbox()
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        self.link_list = list(self.graph_i.nodes)
        self.edge_maps = np.zeros((self.num_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_links,), dtype=int)
        self.link_mapping()
        if cf_radius > 0.5:
            self.add_conflict_relations(cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        self.clear_all_flows()
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        if not self.trace:
            self.delivery = sparse.COO(np.zeros((self.num_nodes, self.num_nodes, self.num_nodes), dtype=float))

    def random_walk(self, ss=0.1, n=10):
        disconnected = True
        while disconnected:
            mask = np.random.choice(np.arange(0, self.num_nodes), size=n, replace=False)
            d_pos = np.random.normal(0, ss, size=(n, 2))
            pos_c_np = self.pos_c_np
            pos_c_np[mask, :] += d_pos
            b_min = np.min(self.box)
            b_max = np.max(self.box)
            pos_c_np = pos_c_np.clip(b_min, b_max)
            d_mtx = distance_matrix(pos_c_np, pos_c_np)
            adj_mtx = np.zeros([self.num_nodes, self.num_nodes], dtype=int)
            adj_mtx[d_mtx <= 1.0] = 1
            np.fill_diagonal(adj_mtx, 0)
            graph_c = nx.from_numpy_array(adj_mtx)
            self.connected = nx.is_connected(graph_c)
            disconnected = not self.connected
        return graph_c, pos_c_np

    def topology_update(self, graph_c, pos_c_np):
        self.graph_c = graph_c
        self.node_positions(pos_c_np)
        self.graph_i = nx.line_graph(self.graph_c)  # Conflict graph
        self.adj_c = nx.adjacency_matrix(self.graph_c)
        self.num_links = len(self.graph_i.nodes)
        link_list_old = self.link_list
        self.link_list = list(self.graph_i.nodes)
        new_links_map = np.zeros((self.num_links,), dtype=int)
        for i in range(self.num_links):
            e0, e1 = self.link_list[i]
            if (e0, e1) in link_list_old:
                j = link_list_old.index((e0, e1))
            elif (e1, e0) in link_list_old:
                j = link_list_old.index((e1, e0))
            else:
                j = -1
            new_links_map[i] = j
        self.edge_maps = np.zeros((self.num_links,), dtype=int)
        self.edge_maps_rev = np.zeros((self.num_links,), dtype=int)
        self.link_mapping()
        if self.cf_radius > 0.5:
            self.add_conflict_relations(self.cf_radius)
        else:
            self.adj_i = nx.adjacency_matrix(self.graph_i)
        self.mean_conflict_degree = np.mean(self.adj_i.sum(axis=0))
        self.W = np.zeros((self.num_links, self.T))
        self.WSign = np.ones((self.num_links, self.T))
        self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
        self.link_comd_cnts = np.zeros((self.num_links, self.num_nodes))
        return new_links_map

    class Flow:
        def __init__(self, source_node, arrival_rate, dest_node):
            self.source_node = source_node
            self.arrival_rate = arrival_rate
            self.dest_node = dest_node
            self.cut_off = -1

    def node_positions(self, pos):
        if pos is None:
            pos_file = os.path.join('..', 'pos', "graph_c_pos_{}.p".format(self.case_name))
            if not os.path.isfile(pos_file):
                pos_c = nx.spring_layout(self.graph_c)
                with open(pos_file, 'wb') as fp:
                    pickle.dump(pos_c, fp, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(pos_file, 'rb') as fp:
                    pos_c = pickle.load(fp)
        elif isinstance(pos, str) and pos == 'new':
            pos_c = nx.spring_layout(self.graph_c)
        elif isinstance(pos, np.ndarray):
            pos_c = dict(zip(list(range(self.num_nodes)), pos))
        else:
            raise ValueError("unsupported pos format in backpressure object initialization")
        self.pos_c = pos_c

    def bbox(self):
        pos_c = np.zeros((self.num_nodes, 2))
        for i in range(self.num_nodes):
            pos_c[i, :] = self.pos_c[i]
        self.pos_c_np = pos_c
        return [np.amin(pos_c[:,0])-0.05, np.amax(pos_c[:,1])+0.05, np.amin(pos_c[:,1])-0.12, np.amax(pos_c[:,1])+0.05]

    def add_conflict_relations(self, cf_radius):
        """
        Adding conflict relationship between links whose nodes are within cf_radius * median_link_distance
        :param cf_radius: multiple of median link distance
        :return: None (modify self.adj_i, and self.graph_i inplace)
        """
        pos_c_vec = np.zeros((self.num_nodes, 2))
        for key, item in self.pos_c.items():
            pos_c_vec[key, :] = item
        dist_mtx = distance_matrix(pos_c_vec, pos_c_vec)
        rows, cols = np.nonzero(self.adj_c)
        link_dist = dist_mtx[rows, cols]
        median_dist = np.nanmedian(link_dist)
        intf_dist = cf_radius * median_dist
        for link in self.link_list:
            src, dst = link
            intf_nbs_s, = np.where(dist_mtx[src, :] < intf_dist)
            intf_nbs_d, = np.where(dist_mtx[dst, :] < intf_dist)
            intf_nbs = np.union1d(intf_nbs_s,intf_nbs_d)
            for v in intf_nbs:
                _, nb2hop = np.nonzero(self.adj_c[v])
                for u in nb2hop:
                    if {v, u} == {src, dst}:
                        continue
                    elif (v, u) in self.link_list:
                        self.graph_i.add_edge((v, u), (src, dst))
                    elif (u, v) in self.link_list:
                        self.graph_i.add_edge((u, v), (src, dst))
                    else:
                        pass
                        # raise RuntimeError("Something wrong with adding conflicting edge")
        self.adj_i = nx.adjacency_matrix(self.graph_i)

    def link_mapping(self):
        # Mapping between links in connectivity graph and nodes in conflict graph
        j = 0
        for e0, e1 in self.graph_c.edges:
            try:
                i = self.link_list.index((e0, e1))
            except:
                i = self.link_list.index((e1, e0))
            self.edge_maps[j] = i
            self.edge_maps_rev[i] = j
            j += 1

    def add_flow(self, src, dst, rate=2, cutoff=-1):
        fi = self.Flow(src, rate, dst)
        if 0 < cutoff < self.T:
            fi.cut_off = int(cutoff)
        else:
            fi.cut_off = self.T
        self.flows.append(fi)
        self.src_nodes.append(src)
        self.dst_nodes.append(dst)
        self.num_flows = len(self.flows)

    def clear_all_flows(self):
        self.flows = []
        self.num_flows = 0
        self.src_nodes = []
        self.dst_nodes = []

    def flows_init(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flows_arrivals = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)
        np.random.seed(self.seed)
        # T = self.T
        # if 0 < cutoff < self.T:
        #     T = int(cutoff)
        for fidx in range(self.num_flows):
            arrival_rate = self.flows[fidx].arrival_rate
            T = int(self.flows[fidx].cut_off)
            self.flows_arrivals[fidx, 0:T] = np.random.poisson(arrival_rate, size=(T,))

    def flows_reset(self):
        self.flows_sink_departures = np.zeros((self.num_flows, self.T), dtype=int)
        self.flow_pkts_in_network = np.zeros((self.num_flows, self.T), dtype=int)

    def links_init(self, rates, std=2):
        if hasattr(rates, '__len__'):
            assert len(rates) == self.num_links
            stds = std*np.ones_like(rates)
        else:
            stds = std
        link_rates = np.zeros((self.num_links, self.T))
        for t in range(self.T):
            link_rates[:, t] = np.clip(np.random.normal(rates, stds), 0, rates + 3*std)
        self.link_rates = np.round(link_rates)

    def queues_init(self):
        # Initialize system state
        self.queue_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.W = np.zeros((self.num_links, self.T))
        self.WSign = np.ones((self.num_links, self.T))
        self.opt_comd_mtx = -np.ones((self.num_links, self.T), dtype=int)
        self.link_comd_cnts = np.zeros((self.num_links, self.num_nodes))
        if self.trace:
            self.backlog = {}
            for i in range(self.num_nodes):
                backlog_i = {}
                for j in range(self.num_nodes):
                    qi = queue.Queue()
                    # qi = collections.deque()
                    backlog_i[j] = qi
                self.backlog[i] = backlog_i
        self.queue_lengths = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_t0 = np.zeros((self.num_nodes, self.num_nodes))
        self.HOL_delay = np.zeros((self.num_nodes, self.num_nodes))
        self.SJT_delay = np.zeros((self.num_nodes, self.num_nodes))

    def pheromone_init(self, decay=0.97, unit=0.01):
        self.phmns_decay = decay
        self.phmns_unit = unit
        self.pheromones = np.zeros((self.num_links, self.num_nodes), dtype=float)
        self.queue_matrix_exp = np.zeros_like(self.queue_matrix)
        self.phmns_exp = 1 + (1-decay)

    def bias_diff(self, bias_matrix):
        link_bias = np.zeros((self.num_links, self.num_nodes), dtype=float)
        for lidx in range(self.num_links):
            src, dst = self.link_list[lidx]
            bdiff = bias_matrix[src, :] - bias_matrix[dst, :]
            link_bias[lidx, :] = bdiff
        return link_bias

    def pkt_arrival(self, t):
        for fidx in range(self.num_flows):
            flow = self.flows[fidx]
            src = flow.source_node
            dst = flow.dest_node
            self.queue_matrix[src, dst] += self.flows_arrivals[fidx, t]
            self.queue_lengths[src, src] += self.flows_arrivals[fidx, t]
            self.queue_matrix_exp[src, dst] += self.flows_arrivals[fidx, t]
            for i in range(self.flows_arrivals[fidx, t]):
                self.backlog[src][dst].put((t, t))

    def update_HOL_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        self.HOL_delay[src][cmd] = 0
                    else:
                        pkt = self.backlog[src][cmd].queue[0]
                        t0, t1 = pkt
                        self.HOL_t0[src][cmd] = t0
                        self.HOL_delay[src][cmd] = t - t1

    def update_SJT_matrix(self, t):
        '''should be run after packet arrivals'''
        if self.trace:
            for src in range(self.num_nodes):
                for cmd in self.dst_nodes:
                    self.SJT_delay[src][cmd] = 0
                    if self.backlog[src][cmd].empty() or (src == cmd):
                        pass
                    else:
                        for pkt in self.backlog[src][cmd].queue:
                            t0, t1 = pkt
                            self.SJT_delay[src][cmd] += t - t1

    def commodity_selection(self, queue_mtx, mbp=0.0, link_phmn=None):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        comds = -np.ones((self.num_links,), dtype=int)
        j = 0
        for link in self.link_list:
            wts_link = queue_mtx[link[0], self.dst_nodes] - queue_mtx[link[1], self.dst_nodes]
            directions = np.sign(wts_link)
            # find out the source nodes
            ql_src_vec = np.where(directions > 0.0,
                                  self.queue_matrix[link[0], self.dst_nodes],
                                  self.queue_matrix[link[1], self.dst_nodes])
            # create a mask that source nodes has more than 1 packet to transmit
            ql_mask = np.where(ql_src_vec > 0.1, np.ones_like(self.dst_nodes), np.zeros_like(self.dst_nodes))
            if link_phmn is None:
                wts_link = np.multiply(wts_link, ql_mask)
            else:
                wts_link = np.multiply(wts_link + link_phmn[j, self.dst_nodes], ql_mask)
            cmd = np.argmax(abs(wts_link))
            W_sign[j] = np.sign(wts_link[cmd])
            W_amp[j] = np.amax([abs(wts_link[cmd]) - mbp, 0])
            comds[j] = self.dst_nodes[cmd] if np.amax(abs(wts_link)) > 0.0 else -1
            # if W_sign[j] == 1:
            #     ql_src = self.queue_matrix[link[0], self.dst_nodes[cmd]]
            # else:
            #     ql_src = self.queue_matrix[link[1], self.dst_nodes[cmd]]
            # comds[j] = self.dst_nodes[cmd] if (np.amax(abs(wts_link)) > 0 and ql_src > 0) else -1
            j += 1
        return W_amp, W_sign, comds

    def commodity_selection_old(self, queue_mtx, mbp=0.0):
        W_amp = np.zeros((self.num_links,), dtype=float)
        W_sign = np.ones((self.num_links,), dtype=float)
        comds = -np.ones((self.num_links,), dtype=int)
        j = 0
        for link in self.link_list:
            wts_link = queue_mtx[link[0], self.dst_nodes] - queue_mtx[link[1], self.dst_nodes]
            comd = np.argmax(abs(wts_link))
            W_sign[j] = np.sign(wts_link[comd])
            W_amp[j] = np.amax([abs(wts_link[comd]) - mbp, 0])
            # comds[j] = self.dst_nodes[comd] if np.amax(abs(wts_link)) > 0 else -1
            if W_sign[j] == 1:
                ql_src = self.queue_matrix[link[0], self.dst_nodes[comd]]
            else:
                ql_src = self.queue_matrix[link[1], self.dst_nodes[comd]]
            comds[j] = self.dst_nodes[comd] if (np.amax(abs(wts_link)) > 0 and ql_src > 0) else -1
            j += 1
        return W_amp, W_sign, comds

    def scheduling(self, weights):
        keep_index = np.argwhere(weights > 0.0)
        wts_postive = weights[keep_index]
        # graph_small = self.graph_i
        adj = self.adj_i[keep_index.flatten(), :]
        adj = adj[:, keep_index.flatten()]
        mwis, total_wt = local_greedy_search(adj, wts_postive)
        solu = list(mwis)
        solu = keep_index[solu].flatten().tolist()
        return solu

    def transmission(self, t, mwis):
        """
        Matrix formed transmission, It takes 0.597 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """
        dsts = -np.ones((len(mwis),), dtype=int)
        srcs = -np.ones_like(dsts)
        schs = -np.ones_like(dsts)
        for idx in range(len(mwis)):
            link = mwis[idx]
            if self.WSign[link, t] < 0:
                dsts[idx] = self.link_list[link][0]
                srcs[idx] = self.link_list[link][1]
            elif self.WSign[link, t] > 0:
                srcs[idx] = self.link_list[link][0]
                dsts[idx] = self.link_list[link][1]
            else:
                continue
            schs[idx] = link
        schs = schs[schs != -1]
        dsts = dsts[dsts != -1]
        srcs = srcs[srcs != -1]
        opt_comds = self.opt_comd_mtx[schs, t]
        num_pkts = np.minimum(self.queue_matrix[srcs, opt_comds], self.link_rates[schs, t])
        if self.trace:
            for idx in range(len(mwis)):
                src = srcs[idx]
                dst = dsts[idx]
                num = num_pkts[idx]
                cmd = opt_comds[idx]
                if cmd == -1:
                    continue
                elif dst == cmd:
                    fidx = self.dst_nodes.index(cmd)
                    self.flows_sink_departures[fidx, t] = num
                for i in range(int(num)):
                    pkt = self.backlog[src][cmd].get_nowait()
                    if pkt is None:
                        raise RuntimeError("Backlog error node: {}, commodity: {}".format(src, cmd))
                    t0, t1 = pkt
                    self.backlog[dst][cmd].put((t0, t))
        self.queue_matrix_exp = self.queue_matrix_exp * self.phmns_exp
        queue_exp_per_pkt = np.nan_to_num(np.divide(self.queue_matrix_exp[srcs, opt_comds], self.queue_matrix[srcs, opt_comds]), nan=0.0)
        self.queue_matrix_exp[dsts, opt_comds] += num_pkts
        self.queue_matrix_exp[srcs, opt_comds] -= queue_exp_per_pkt * num_pkts
        self.queue_matrix_exp[self.queue_matrix_exp < 0.5] = 0.0
        self.queue_matrix[dsts, opt_comds] += num_pkts
        self.queue_matrix[srcs, opt_comds] -= num_pkts

        if not self.trace:
            coords = np.vstack([srcs, dsts, opt_comds])
            coo_pkts = sparse.COO(coords=coords, data=num_pkts, shape=(self.num_nodes, self.num_nodes, self.num_nodes))
            self.delivery += coo_pkts

        self.link_comd_cnts[schs, opt_comds] += num_pkts
        self.pheromones = self.pheromones * self.phmns_decay
        self.pheromones[schs, opt_comds] += self.phmns_unit * np.multiply(num_pkts, self.WSign[mwis, t])
        # there are shadow commodities in SP-bias (0 packets to transmit)
        sink_true = np.logical_and(opt_comds==dsts, num_pkts>0.1)
        sink_dsts = dsts[sink_true]
        if len(sink_dsts) > 0:
            self.queue_matrix[sink_dsts, sink_dsts] = 0
            self.queue_matrix_exp[sink_dsts, sink_dsts] = 0
            if not self.trace:
                fidxs = np.zeros_like(sink_dsts)
                for sidx in range(len(sink_dsts)):
                    fidx, = np.where(self.dst_nodes == sink_dsts[sidx])
                    if len(fidx) > 0:
                        fidxs[sidx] = fidx[0]
                self.flows_sink_departures[fidxs, t] = num_pkts[sink_true]

    def transmission_old(self, t, mwis):
        """
        Old transmission, It takes 0.664 seconds to run 100 time slots on graph (15) seed 3
        :param t: time
        :param mwis: list of scheduled links
        :return:
        """
        for edge in mwis:
            if self.WSign[edge, t] < 0:
                dst = self.link_list[edge][0]
                src = self.link_list[edge][1]
            elif self.WSign[edge, t] > 0:
                src = self.link_list[edge][0]
                dst = self.link_list[edge][1]
            else:
                continue

            opt_comd = self.opt_comd_mtx[edge, t]
            num_pkts = min(self.queue_matrix[src, opt_comd], self.link_rates[edge][t])
            self.queue_matrix[dst, opt_comd] += num_pkts
            self.queue_matrix[src, opt_comd] -= num_pkts
            self.link_comd_cnts[edge, opt_comd] += num_pkts
            if opt_comd == dst:
                self.queue_matrix[dst, opt_comd] = 0
                for fidx in range(self.num_flows):
                    if self.flows[fidx].dest_node == dst:
                        self.flows_sink_departures[fidx, t] = num_pkts

    def update_bias_mean(self, bias_matrix):
        # step 1: find out neighbors, construct an out adj matrix
        out_adj = np.zeros((self.num_nodes, self.num_nodes))
        bias_matrix_new = np.copy(bias_matrix)
        for cmdty in self.dst_nodes:
            for idx_link in range(self.num_links):
                e0, e1 = self.link_list[idx_link]
                val = self.pheromones[idx_link, cmdty]
                if val > 0:
                    out_adj[e0, e1] = abs(val)
                elif val < 0:
                    out_adj[e1, e0] = abs(val)
                else:
                    pass
            out_adj = out_adj / np.linalg.norm(out_adj, ord=1, axis=1, keepdims=True)
            # step 2: update bias
            tmp = np.dot(out_adj, bias_matrix[cmdty]+1)
            bias_matrix_new[~np.isnan(tmp), cmdty] = tmp[~np.isnan(tmp)]
            bias_matrix_new[cmdty, cmdty] = 0
        return bias_matrix_new

    def update_bias(self, bias_matrix, delay_mtx):
        # step 1: find out neighbors, construct an out adj matrix
        bias_matrix_new = np.copy(bias_matrix)
        for v in range(self.num_nodes):
            _, nb_set = np.nonzero(self.adj_c[v])
            sp_v = (bias_matrix[nb_set, :] + delay_mtx[nb_set, v:v+1]).min(axis=0)
            bias_matrix_new[v, :] = np.minimum(sp_v, bias_matrix[v, :])
        return bias_matrix_new

    def estimate_delay(self, num_itn=20):
        self.queue_lengths = self.queue_lengths / float(self.T)
        out_pkts = np.zeros((self.num_nodes, self.num_flows))
        delay_local = np.inf * np.ones_like(out_pkts)
        # delta_n2c = np.ones_like(out_pkts)
        route_prob = {}

        for node in range(self.num_nodes):
            _, nb_list = np.nonzero(self.adj_c[node])
            pkts_nbs = self.delivery[node, :, :]
            pkts_nbs = pkts_nbs[list(nb_list), :]
            pkts_nbs = pkts_nbs[:, self.dst_nodes].todense()
            out_pkts[node, :] = pkts_nbs.sum(axis=0)
            edge_prob = pkts_nbs / out_pkts[node, :]
            route_prob[node] = (edge_prob, nb_list)
        delay_local[:, :] = float(self.T) * self.queue_lengths[:, self.dst_nodes] / out_pkts
        delay_local[delay_local == np.inf] = self.T
        delay_local[delay_local == np.nan] = 0
        for i in range(self.num_flows):
            delay_local[self.dst_nodes[i], i] = 0

        delta_n2c = delay_local.copy()
        diff_vec  = np.zeros((num_itn,))
        for itn in range(num_itn):
            delta_n2c_prev = delta_n2c.copy()
            for i in range(self.num_nodes):
                edge_prob, nb_list = route_prob[i]
                delta_nxt = np.nansum(np.multiply(edge_prob, delta_n2c[nb_list, :]), axis=0)
                delta_n2c[i, :] = delay_local[i, :] + delta_nxt
            for i in range(self.num_flows):
                delta_n2c[self.dst_nodes[i], i] = 0
            delta_diff = np.mean(delta_n2c - delta_n2c_prev)
            diff_vec[itn] = delta_diff

        return delta_n2c, diff_vec

    def collect_delay(self):
        flows_in = self.flows_arrivals.sum(axis=1)
        flows_out = np.zeros((self.num_flows,), dtype=int)
        flows_delay = np.zeros((self.num_flows,))
        flows_delay_est = np.zeros((self.num_flows,))
        flows_delay_raw = []
        flows_undeliver = []
        for fidx in range(self.num_flows):
            flow = self.flows[fidx]
            src = flow.source_node
            dst = flow.dest_node
            flows_out[fidx] = len(self.backlog[dst][dst].queue)
            delay_per_pkt = np.zeros((flows_out[fidx],))
            for i in range(flows_out[fidx]):
                t0, t1 = self.backlog[dst][dst].queue[i]
                delay_per_pkt[i] = float(t1-t0)
            flows_delay[fidx] = np.mean(delay_per_pkt)
            flows_delay_raw.append(delay_per_pkt)
            delay_undelivered = []
            for i in range(self.num_nodes):
                if i == flow.dest_node:
                    continue
                for idx in range(len(self.backlog[i][dst].queue)):
                    t0, t1 = self.backlog[i][dst].queue[idx]
                    delay_undelivered.append(self.T - t0)
            delay_undelivered = np.array(delay_undelivered)
            delay_all_pkts = np.concatenate((delay_per_pkt, delay_undelivered), axis=0)
            flows_delay_est[fidx] = np.nanmean(delay_all_pkts)
            flows_undeliver.append(delay_undelivered)

        return flows_in, flows_out, flows_delay, flows_delay_raw, flows_delay_est, flows_undeliver

    def plot_routes(self, delays, opt, with_labels=True, fdisp=-1):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        for f_show in range(len(self.flows)):
            if 0 <= fdisp < len(self.flows):
                if fdisp != f_show:
                    continue
            f_cnts = self.link_comd_cnts[self.edge_maps, self.flows[f_show].dest_node]
            # weights = 1 + 10 * f_cnts / (np.amax(f_cnts)+0.000001)
            weights = 1 + np.float_power(f_cnts, 0.33)
            vis_network(
                self.graph_c,
                self.src_nodes[f_show:f_show+1],
                self.dst_nodes[f_show:f_show+1],
                self.pos_c,
                weights,
                delay_f[:, f_show],
                with_labels
            )
            fig_name = "flow_routes_visual_{}_f{}_s{}_d{}_cf{:.0f}_opt{}.png".format(
                self.case_name, f_show,
                self.flows[f_show].source_node,
                self.flows[f_show].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            ax = plt.gca()
            ax.set_xlim(bbox[0:2])
            ax.set_ylim(bbox[2:4])
            # plt.tight_layout(pad=-0.1)
            plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            print("Flow {} plot saved to {}".format(f_show, fig_name))

    def plot_pheromones(self, delays, opt, with_labels=True):
        delay_f = np.nan_to_num(delays)
        bbox = self.bbox()
        for f_show in range(len(self.flows)):
            f_cnts = np.abs(self.pheromones[self.edge_maps, self.flows[f_show].dest_node])
            weights = 1 + 10 * f_cnts / (np.amax(f_cnts)+0.000001)
            vis_network(
                self.graph_c,
                self.src_nodes[f_show:f_show+1],
                self.dst_nodes[f_show:f_show+1],
                self.pos_c,
                weights,
                delay_f[:, f_show],
                with_labels
            )
            fig_name = "flow_pheromone_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, f_show,
                self.flows[f_show].source_node,
                self.flows[f_show].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            ax = plt.gca()
            ax.set_xlim(bbox[0:2])
            ax.set_ylim(bbox[2:4])
            # plt.tight_layout(pad=-0.1)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close()
            print("Flow {} plot saved to {}".format(f_show, fig_name))

    def plot_delay(self, delay_n2c, opt):
        for f_show in range(self.num_flows):
            node_colors = ['y' for node in range(self.num_nodes)]
            node_sizes = 10*delay_n2c[:, f_show]
            node_colors[self.src_nodes[f_show]] = 'g'
            node_colors[self.dst_nodes[f_show]] = 'b'
            node_sizes[self.dst_nodes[f_show]] = 400
            ax = nx.draw(
                self.graph_c,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                pos=self.pos_c)
            fig_name = "flow_delay_visual_{}_f{}_s{}_d{}_cf{:.1f}_opt{}.png".format(
                self.case_name, f_show,
                self.flows[f_show].source_node,
                self.flows[f_show].dest_node,
                self.cf_radius,
                opt)
            fig_name = os.path.join("..", "fig", fig_name)
            plt.savefig(fig_name, dpi=300)
            plt.close()
            print("Flow {} plot saved to {}".format(f_show, fig_name))

    def plot_metrics(self, opt):
        arrivals = np.sum(self.flows_arrivals, axis=0)
        pkts_in_network = np.sum(self.flow_pkts_in_network, axis=0)
        departures = np.sum(self.flows_sink_departures, axis=0)

        plt.plot(arrivals)
        plt.plot(departures)
        plt.plot(pkts_in_network)

        plt.suptitle('Departures, Arrivals, and Current amount pkts in network')
        plt.xlabel('T')
        plt.ylabel('the number of packages')
        plt.legend(['Exogenous arrivals', 'Sink departures', 'Pkts in network'], loc='upper right')
        fig_name = "flow_packets_arrivals_per_timeslot_{}_cf{:.1f}_opt_{}.png".format(self.case_name, self.cf_radius, opt)
        fig_name = os.path.join("..", "fig", fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close()
        print("Metrics plot saved to {}".format(fig_name))
        return arrivals, pkts_in_network, departures


def main(args):
    # Configuration
    NUM_NODES = 15
    # LAMBDA = 1 # we are not use it
    T = 500
    link_rate = 5*4
    cf_radius = 0.0 # relative conflict radius based on physical distance model

    opt = int(args[0])
    seed = 3

    # Create fig folder if not exist
    if not os.path.isdir(os.path.join("..", "fig")):
        os.mkdir(os.path.join("..", "fig"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "pos")):
        os.mkdir(os.path.join("..", "pos"))
    # Create pos folder if not exist
    if not os.path.isdir(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    start_time = time.time()
    # bp_env = Backpressure(NUM_NODES, T, seed)
    bp_env = Backpressure(NUM_NODES, T, seed, cf_radius=cf_radius, trace=True)

    #f0 = Flow(3, 2, 6)
    #f1 = Flow(1, 5, 5)
    bp_env.add_flow(3, 6, rate=2)
    bp_env.add_flow(1, 7, rate=5)
    bp_env.flows_init()

    bp_env.links_init(link_rate)
    bp_env.queues_init()

    # shortest_paths = np.zeros((NUM_NODES, NUM_NODES))
    shortest_paths = all_pairs_shortest_paths(bp_env.graph_c)

    bias_matrix = np.zeros_like(bp_env.queue_matrix)

    logfile = os.path.join("..", "out", "Output_{}_opt_{}.txt".format(bp_env.case_name, opt))
    with open(logfile, "a") as f:
        print("Edges:", file=f)
        print(bp_env.graph_i.nodes(), file=f)
        print("Link Rates:", file=f)
        print(bp_env.link_rates, file=f)

    print("Init graph('{}') in {:.3f} seconds".format(bp_env.case_name, time.time() - start_time),
          ": conflict radius {}, degree {:.2f}".format(bp_env.cf_radius, bp_env.mean_conflict_degree))
    start_time = time.time()

    for t in range(bp_env.T):
        bp_env.pkt_arrival(t)

        # Bias computation
        if opt == 1:  # shortest path bias
            bias_matrix = shortest_paths
        elif opt == 0:
            pass
        else:
            raise ValueError("unsupported opt {}".format(opt))

        # Commodity and W computation
        W_amp, W_sign, C_opt = bp_env.commodity_selection(bp_env.queue_matrix + bias_matrix)
        W_amp[C_opt == -1] = 0.0
        bp_env.W[:, t] = W_amp
        bp_env.WSign[:, t] = W_sign
        bp_env.opt_comd_mtx[:, t] = C_opt

        # Greedy Maximal Scheduling & Transmission
        mwis = bp_env.scheduling(bp_env.W[:, t] * bp_env.link_rates[:, t])
        bp_env.transmission(t, mwis)
        # bp_env.transmission_old(t, mwis)

        for fidx in range(bp_env.num_flows):
            bp_env.flow_pkts_in_network[fidx, t] = np.sum(bp_env.queue_matrix[:, bp_env.flows[fidx].dest_node])

    print("Main loop {} time slots in {:.3f} seconds".format(T, time.time() - start_time))
    bp_env.plot_metrics(opt)

    start_time = time.time()
    # delay_n2c, diff_vec = bp_env.estimate_delay()
    # print("Estimating delay in {:.3f} seconds".format(time.time() - start_time))
    # bp_env.plot_routes(delay_n2c, opt)
    cnt_in, cnt_out, delay_e2e, delay_e2e_raw, delay_est, undeliver = bp_env.collect_delay()
    print("Estimating delay in {:.3f} seconds".format(time.time() - start_time))
    delay_est_vec = np.ones((bp_env.num_nodes, bp_env.num_flows))
    delay_est_vec = np.multiply(np.reshape(delay_est, (1, bp_env.num_flows)), delay_est_vec)
    bp_env.plot_routes(delay_est_vec, opt)

    print("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
