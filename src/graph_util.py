# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.spatial import distance_matrix


# Utility functions
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def simple_polynomials(adj, k):
    """Calculate polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # laplacian = symmetric_graph_laplacian(adj)

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(laplacian)

    for i in range(2, k+1):
        t_new = t_k[-1]*laplacian
        t_k.append(t_new)

    return sparse_to_tuple(t_k)


