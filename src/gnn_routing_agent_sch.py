# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function
import sys
import os
import time
import datetime
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import random
from multiprocessing import Queue
from collections import deque
from copy import deepcopy
import networkx as nx
# Tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, schedules
from tensorflow.keras.regularizers import l2
# Spektral
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ChebConv
from spektral.utils import sp_matrix_to_sp_tensor, reorder
from spektral.transforms import LayerPreprocess
# Graph utility
from graph_util import *
import warnings
warnings.filterwarnings('ignore')

# input flags
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('datapath', '../data_100', 'input data path.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('out', '../out', 'output data path.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_integer('T', 100, 'Number of time slots.')
flags.DEFINE_string('training_set', 'BAm2', 'Name of training dataset')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_decay', 1.0, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('num_layer', 5, 'number of layers.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('epsilon', 1.0, 'initial exploration rate')
flags.DEFINE_float('epsilon_min', 0.001, 'minimal exploration rate')
flags.DEFINE_float('epsilon_decay', 0.985, 'exploration rate decay per replay')
flags.DEFINE_float('gamma', 1.0, 'gamma')
flags.DEFINE_integer('batch', 10, 'batch size.')


# Agent
class GDPGAgent:
    def __init__(self, input_flags, memory_size=5000):
        # super(GDPGAgent, self).__init__(input_flags, memory_size)
        self.flags = input_flags
        self.learning_rate = self.flags.learning_rate
        self.n_node_features = 1
        self.n_edge_features = 1
        self.output_size = 2
        self.max_degree = 1
        self.num_supports = 1 + self.max_degree
        self.l2_reg = self.flags.weight_decay
        self.epsilon = self.flags.epsilon
        self.model = self._build_model()
        self.memory = deque(maxlen=memory_size)
        self.memory_crt = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.mse = MeanSquaredError()
        self.name = "ChebConv"
        # self.log_init()

    def _build_model(self):
        # Neural Net for Actor Model
        x_in = Input(shape=(self.n_node_features,), dtype=tf.float64, name="x_in")
        e_in = Input(shape=(self.n_edge_features,), dtype=tf.float64, name="e_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        gc_l = x_in
        # gc_l = tf.ones_like(x_in, dtype=tf.float64, name="ones_in")
        for l in range(self.flags.num_layer):
            if l < self.flags.num_layer - 1:
                act = "leaky_relu"
                output_dim = 32
            else:
                act = "softmax"
                output_dim = self.output_size
            do_l = Dropout(self.flags.dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                output_dim, K=self.num_supports, activation=act,
                kernel_regularizer=l2(self.l2_reg),
                use_bias=True,
                dtype='float64'
            )([do_l, a_in])

        # Build model
        model = Model(inputs=[x_in, a_in], outputs=gc_l)
        if self.flags.learning_decay == 1.0:
            self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        else:
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=100,
                decay_rate=self.flags.learning_decay)
            self.optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.summary()
        return model

    def load(self, name):
        ckpt = tf.train.latest_checkpoint(name)
        if ckpt:
            self.model.load_weights(ckpt)
            print('Actor loaded ' + ckpt)

    def save(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def makestate(self, adj, node_features):
        # reduced_nn = node_features.shape[0]
        support = simple_polynomials(adj, self.max_degree)
        state = {"node_features": node_features,
                 "support": support[1]}
        return state

    def memorize(self, grad, loss, reward):
        self.memory.append((grad.copy(), loss, reward))

    def predict(self, state):
        x_in = tf.convert_to_tensor(state["node_features"], dtype=tf.float64)
        coord, values, shape = state["support"]
        a_in = tf.sparse.SparseTensor(coord, values, shape)
        # a_in = sp_matrix_to_sp_tensor(state["support"])
        act_values = self.model([x_in, a_in])
        return act_values

    def act(self, state):
        high_dimensional_action = self.predict(state)
        return high_dimensional_action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return float('NaN')
        self.reward_mem.clear()
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for grad, loss, _ in minibatch:
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
            losses.append(loss)

        self.memory.clear()
        if self.epsilon > self.flags.epsilon_min:
            self.epsilon *= self.flags.epsilon_decay
        return np.nanmean(losses)

    def foo_train(self, adj_0, lr_0, ind_vec, train=False):
        adj = adj_0.copy()
        nn  = lr_0.shape[0]
        # node_features = np.reshape(lr_0, (nn, 1))
        node_features = np.ones((nn, 1), dtype=float)
        ind_vec_comp = np.ones((nn, 2), dtype=float)
        ind_vec_comp[:, 0] = ind_vec.flatten()
        ind_vec_comp[:, 1] = 1.0-ind_vec.flatten()

        # GNN
        with tf.GradientTape() as g:
            g.watch(self.model.trainable_weights)
            state = self.makestate(adj, node_features)
            bias_ts = self.act(state)

            if train:
                regularization_loss = tf.cast(tf.reduce_sum(self.model.losses), dtype=tf.float64)
                y_target = tf.convert_to_tensor(ind_vec_comp, dtype=tf.float64)
                loss_value = tf.sqrt(self.mse(y_target, bias_ts)) + regularization_loss
                gradients = g.gradient(loss_value, self.model.trainable_weights)
                self.memorize(gradients, loss_value.numpy(), 0.0)
        return state, bias_ts.numpy()[:, 0]

    def topology_encode(self, adj_0, lr_0):
        adj = adj_0.copy()
        nn  = lr_0.shape[0]
        # node_features = np.reshape(lr_0, (nn, 1))
        node_features = np.ones((nn, 1), dtype=float)

        # GCN
        state = self.makestate(adj, node_features)
        bias_ts = self.act(state)

        return bias_ts.numpy()[:, 0]


# use gpu 0
# os.environ['CUDA_VISIBLE_DEVICES']=str(0)
#
# # Initialize session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
