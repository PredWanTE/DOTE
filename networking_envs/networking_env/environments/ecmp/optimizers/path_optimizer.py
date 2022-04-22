import numpy as np
import networkx as nx
import tqdm
from itertools import islice
import joblib
from copy import deepcopy
from networking_env.environments.consts import HistoryConsts, ActionConsts
from networking_env.environments.ecmp import utils
from networking_env.environments.ecmp.optimizers.optimizer import Optimizer
from networking_env.environments.ecmp.optimizers import consts as OptConsts
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, lil_matrix


class PathOptimizer(Optimizer):

    def __init__(self, adj, c_vec, w_adj, props=None):
        super(PathOptimizer, self).__init__(adj, c_vec, w_adj, props)
        
    def _paths_from_file1(self):
        '''
        file format is assumed to be : i,j:i,v_1,v_2,...,v_n,j
        '''
        import glob
        graph_folder = "/".join(self.props.graph_path.split("/")[:-1])
        
        def tsort(k1, k2):
            s1 = int(k1.split("_")[0])
            s2 = int(k2.split("_")[0])
            if s1 == s2:
                d1 = int(k1.split("_")[1].replace(".tunnels",""))
                d2 = int(k2.split("_")[1].replace(".tunnels",""))
                return d1<d2
            else:
                return s1<s2
        
        tunnel_files = sorted( glob.glob(graph_folder+"/tunnels/*.tunnels") ,key=lambda x: x.replace(".tunnels", "") )
         
        pij = defaultdict(list)
        commid = 0
        numpaths = 0
        for src in range(self._num_nodes):
            for dst in range(self._num_nodes):
                if src == dst:
                    continue
                try:
                    tunnel_file = tunnel_files[commid]
                    paths = [l.strip() for l in open(tunnel_file) if l.strip()]
                    for p_ in paths:
                        node_list = list(map(int, p_.split("-")))
                        pij[(src, dst)].append( elf._to_path(node_list))
                        numpaths += 1
                except: import pdb; pdb.set_trace()

    def _paths_from_file(self):
        paths_file = "%s/%s/%s"%(self.props.graph_base_path, self.props.ecmp_topo, self.props.paths_file)
        pij = defaultdict(list)
        pid = 0
        with open(paths_file, 'r') as f:
            lines = sorted(f.readlines())
            lines_dict = {line.split(":")[0] : line for line in lines if line.strip() != ""}
            for src in range(self._num_nodes):
                for dst in range(self._num_nodes):
                    if src == dst:
                        continue
                    try:
                        if "%d %d" % (src, dst) in lines_dict:
                            line = lines_dict["%d %d" % (src, dst)].strip()
                        else:
                            line = [l for l in lines if l.startswith("%d %d:" % (src, dst))]
                            if line == []:
                                continue
                            line = line[0]
                            line = line.strip()
                        if not line: continue
                        i,j = list(map(int, line.split(":")[0].split(" ")))
                        paths = line.split(":")[1].split(",")
                        for p_ in paths:
                            node_list = list(map( int, p_.split("-")))
                            pij[(i, j)].append(self._to_path(node_list))
                            pid += 1
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()
        return pij
    
    def _to_path(self, node_list):
        return [(v1, v2) for v1, v2 in zip(node_list, node_list[1:])]

    def _k_shortest_paths(self, G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    def _paths_from_ksp(self):
        g = nx.from_numpy_matrix(self._adj)
        pij = defaultdict(list)

        print('[+] Finding K shortest paths.')
        with open('tunnels.txt', 'w') as f:
            for src in tqdm.tqdm(range(self._num_nodes)):
                for dst in tqdm.tqdm(range(self._num_nodes)):
                    if src == dst:
                        continue
                    try:
                        all_paths = self._k_shortest_paths(g, src, dst, self.props.num_tunnels)
                        f.write('%d %d:' % (src, dst))
                        tunnel_string = ','.join(['-'.join([str(_) for _ in i]) for i in all_paths])
                        f.write(tunnel_string + '\n')
                        pij[(src, dst)] = [self._to_path(all_paths[i]) for i in range(self.props.num_tunnels)]
                    except:
                        pij[(src, dst)] = [self._to_path(all_paths[i]) for i in range(len(all_paths))]

        return pij

    # k shortest paths using weighted yen's algorithm
    def _paths_from_ksp_weighted(self):
        # get 1/adj to get inverse capacities
        inv_adj = np.reciprocal(self._w_adj)

        # zero out infinity which caused from 1/0 for edges that don't exist
        inv_adj[np.where(inv_adj == np.inf)] = 0

        # add constant
        inv_adj[np.where(inv_adj != 0)] += 200

        # create graph
        g = nx.from_numpy_matrix(inv_adj)
        pij = defaultdict(list)

        print('[+] Finding Weighted K shortest paths.')
        for src in tqdm.tqdm(range(self._num_nodes)):
            for dst in tqdm.tqdm(range(self._num_nodes)):
                if src == dst:
                    continue
                try:
                    all_paths = self._k_shortest_paths(g, src, dst, self.props.num_tunnels, weight='weight')
                    pij[(src, dst)] = [self._to_path(all_paths[i]) for i in range(self.props.num_tunnels)]
                except:
                    pij[(src, dst)] = [self._to_path(all_paths[i]) for i in range(len(all_paths))]

        return pij

    def _paths_from_sp(self):
        g = nx.from_numpy_matrix(self._adj)
        
        pij = defaultdict(list)
        
        with open("tunnels.txt", "w") as f:
            for src in range(self._num_nodes):
                for dst in range(self._num_nodes):
                    if src == dst:
                        continue
                    num_tunnels = self.props.num_tunnels
                    paths = []
                    g_ = g.copy()
                    for _ in range(num_tunnels):
                        if nx.has_path(g_, src, dst):
                            node_list = nx.shortest_path(g_, src, dst)
                            path = self._to_path(node_list)
                            paths.append(path)
                            g_.remove_edges_from(path)
                        else:
                            break
                            import pdb; pdb.set_trace()
                    
                    pij[(src, dst)] = paths
                    f.write('%d %d:' % (src, dst))
                    tunnel_string = ','.join(['-'.join([str(_[0]) for _ in i] + [str(i[-1][1])]) for i in paths]) #todo - add last
                    f.write(tunnel_string + '\n')
        return pij

    def _paths_from_flow(self):
        def traverse(g, used_edges, src, dst):
            path = []
            if src==dst: return path
            else:
                for neigh,v in g[src].items():
                    if (src,neigh) not in used_edges:
                        if v>0:
                            e = (src,neigh)
                            path.append(e)
                            used_edges.add(e)
                            return path+traverse(g, used_edges, neigh, dst)
        
        g = nx.from_numpy_matrix(self._adj)
        nx.set_edge_attributes(g,'capacity',{e:1 for e in g.edges()})
        
        
        pij = defaultdict(list)
        
        for src in range(self._num_nodes):
            for dst in range(self._num_nodes):
                if src==dst: continue
                num_tunnels = self.props.num_tunnels
                paths = []
                flow, g_flow = nx.maximum_flow(g,src,dst)
                assert flow >= num_tunnels
                used_edges = set()
                for _ in range(num_tunnels):
                    # now convert maxflow into edge disjoint paths
                    p = traverse(g_flow,used_edges,src,dst)
                    paths.append(p)
                pij[(src,dst)] = paths
        return pij
    
    def _init(self, adj, c_vec):
        if self.props.paths_from == OptConsts.PATHS_FROM_FILE:
            pij = self._paths_from_file()
        elif self.props.paths_from == OptConsts.PATHS_FROM_K_SHORTEST_PATHS:
            pij = self._paths_from_ksp()
        elif self.props.paths_from == OptConsts.PATHS_FROM_K_WEIGHTED_SHORTEST_PATHS:
            pij = self._paths_from_ksp_weighted()
        elif self.props.paths_from == OptConsts.PATHS_FROM_SHORTEST_PATHS: 
            pij = self._paths_from_sp()    
        elif self.props.paths_from == OptConsts.PATHS_FROM_FLOW: 
            pij = self._paths_from_flow()    
        self._paths_to_edges, self._commodities_paths = self._nodes_to_edges(pij)

        all_paths, _ = self._nodes_to_edges(pij)
        self._paths_to_edges = csr_matrix(all_paths)
        self._num_paths = all_paths.shape[0]
        self._pij = pij
        self._make_paths_params()
        
        self._from_weights = self.props.history_action_type in ActionConsts.ACTIONS_W
        #self._test = self.props.history_action_type in ActionConsts.TEST

    def _make_weights_to_edges(self, paths_to_edges, num_weights):
        weights_to_edges = np.zeros((num_weights, paths_to_edges.shape[1]))
        weights_of_other_edges = np.zeros((paths_to_edges.shape[1], ))
        idx = 0
        for col in range(paths_to_edges.shape[1]):
            if np.any(paths_to_edges[:, col]):
                weights_to_edges[idx, col] = 1
                idx += 1
            else:
                weights_of_other_edges[col] = 1

        return csr_matrix(weights_to_edges), weights_of_other_edges

    def _make_paths_params(self):
        #commodities_to_paths = np.zeros((self._num_nodes * (self._num_nodes - 1),
        #                                 self._num_paths))
        commodities_to_paths = lil_matrix((self._num_nodes * (self._num_nodes - 1), self._num_paths))
        commodities_path_cntr = np.zeros((self._num_nodes * (self._num_nodes - 1), ))
        commid = 0
        pathid = 0
        for src in range(self._num_nodes):
            for dst in range(self._num_nodes):
                if src == dst:
                    continue
                for _ in self._pij[(src, dst)]:
                    commodities_to_paths[commid, pathid] = 1
                    commodities_path_cntr[commid] += 1
                    pathid += 1
                commid += 1 
        
        self._commodities_to_paths = csr_matrix(commodities_to_paths)
        self._commodities_path_counter = commodities_path_cntr

    def _nodes_to_edges1(self, paths):
        paths_ = {}
        paths_arr = []
        for i in range(self._num_nodes):
            paths_[i] = []
            pi = []
            for j in range(self._num_nodes):
                if i == j:
                    continue
                pij = []
                for p in paths[(i, j)]:
                    p_ = [self._edges_map[e] for e in p ]
                    p__ = np.zeros((int(self._num_edges),))
                    for k in p_: p__[k] = 1
                    pij.append(p__)
                pi.append(np.stack(pij))
            try:
                paths_[i] = np.stack(pi)
                paths_arr.append(np.stack(pi))
            except: import pdb; pdb.set_trace()
        return paths_, paths_arr

    def _set_edges_map(self):
        eid = 0
        self._edges_map = {}
        self._edges_map_rev = {}
        for i in range(self._adj.shape[0]):
            for j in range(self._adj.shape[1]):
                if self._adj[i, j] == 1:
                    self._edges_map[(i, j)] = eid
                    self._edges_map_rev[eid] = (i, j)
                    eid += 1
        
        self._edge_to_op_dir = {}
        for id in self._edges_map:
            self._edge_to_op_dir[self._edges_map[id]] = self._edges_map[((id[1],id[0]))]

    def _nodes_to_edges(self, paths):
        paths_arr = []
        commodities = []
        self._path_to_commodity = dict()
        self._path_to_idx = dict()
        cntr = 0
        for i in range(self._num_nodes):
            for j in range(self._num_nodes):
                if i == j:
                    continue
                idx = 0
                for p in paths[(i, j)]:
                    p_ = [self._edges_map[e] for e in p]
                    p__ = np.zeros((int(self._num_edges),))
                    for k in p_:
                        p__[k] = 1
                    paths_arr.append(p__)
                    self._path_to_commodity[cntr] = (i, j)
                    self._path_to_idx[cntr] = idx
                    cntr += 1
                    idx += 1
                    commodities.append((i, j))
        return np.stack(paths_arr), commodities

    def step(self, w, tm, cost_only=True):
        if self._from_weights:
            paths_weight = self._paths_to_edges.dot(w)
        else:
            paths_weight = w

        paths_weight = np.exp(paths_weight)
        commodity_total_weight = self._commodities_to_paths.dot(paths_weight)

        # this will force equal split across missing paths/commodities
        commodity_total_weight[commodity_total_weight <= 0] = \
            self._commodities_path_counter[commodity_total_weight <= 0]
        commodity_total_weight = 1.0 / commodity_total_weight

        # now find the splitting ratios
        paths_over_total = self._commodities_to_paths.T.dot(commodity_total_weight)
        paths_split = np.multiply(paths_weight, paths_over_total)

        # now find demand over the paths
        # this is what costs most
        demand_on_paths = self._commodities_to_paths.multiply(paths_split).T.dot(tm)
        # demand_on_paths = np.multiply(self._commodities_to_paths, paths_split).T.dot(tm)

        # now find congestion
        flow_on_edges = self._paths_to_edges.T.dot(demand_on_paths)
        congestion = flow_on_edges / self._c
        return max(congestion)
