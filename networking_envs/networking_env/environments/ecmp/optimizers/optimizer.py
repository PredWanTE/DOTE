from networking_env.environments.ecmp import utils
import numpy as np


class Optimizer:
    def __init__(self, adj, c_vec, w_adj=None, props=None):
        self.props = props
        self._c = c_vec
        self._adj = adj
        self._w_adj = w_adj
        self._set_edges_map()
        self._set_as(adj)
        self._init(adj, c_vec)
    
    def _init(self, adj, c_vec):
        raise NotImplementedError()
    
    def step(self, action, tm):
        raise NotImplementedError()
    
    def _set_as(self, adj):
        num_edges, a_in, a_out, _ = utils.get_as(adj)
                    
        self._num_edges = num_edges
        self._num_nodes = adj.shape[0]
        self._a_in = a_in
        self._a_out = a_out
        self._zero_diagonal = np.ones_like(adj, dtype=np.float32) - np.eye(self._num_nodes, dtype=np.float32)
        self._adj = adj
    
    def _set_edges_map(self):
        eid = 0
        self._edges_map = {}
        for i in range(self._adj.shape[0]):
            for j in range(self._adj.shape[1]):
                if self._adj[i, j] == 1:
                    self._edges_map[(i, j)] = eid
                    eid += 1
                    
    def direct_q_value(self): pass
    
    def reset(self): pass