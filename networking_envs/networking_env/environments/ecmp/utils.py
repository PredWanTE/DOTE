import numpy as np
from networking_env.environments.consts import EdgeConsts, HistoryConsts, ActionPostProcess
import networkx
from collections import OrderedDict

def get_as(adj):
    num_edges = np.int32(np.sum(adj))
    a_out = np.zeros((adj.shape[0],num_edges))
    a_in = np.zeros((adj.shape[0],num_edges))
    eid = 0
    e_map = {}
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i,j] == 1:
                a_out[i,eid] = 1
                a_in[j,eid] = 1
                e_map[(i,j)] = eid
                eid+=1
    return num_edges, a_in, a_out, e_map

def get_base_graph():
    # init a triangle if we don't get a network graph
    g = networkx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([ (0, 1, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (1, 2, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (2, 0, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (4, 0, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (3, 2, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (3, 4, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (4, 5, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0}),
                        (5, 1, {EdgeConsts.WEIGHT_STR:1, EdgeConsts.CAPACITY_STR:1, EdgeConsts.TTL_FLOW_STR: 0})])

    return g


def load_graph_from_file(g_name):
    # each line is of the from "u,v,capacity" - edges are assumed to be bidirectional
    with open(g_name, 'r') as f:

        links = OrderedDict()
        for line in f:
            line = line.strip()
            if not line: continue
            u, v, cap = map(float, line.strip().split(","))
            links[(u,v)] = cap
        
        nodes = set()
        edges = []
        for (u, v), cap in links.items():
            nodes.add(u)
            nodes.add(v)
            edges.append((u, v, {EdgeConsts.WEIGHT_STR:1.0,
                                 EdgeConsts.CAPACITY_STR:cap,
                                 EdgeConsts.TTL_FLOW_STR: 0}))

            if (v, u) not in links:
                edges.append((v, u, {EdgeConsts.WEIGHT_STR:1.0,
                                     EdgeConsts.CAPACITY_STR:cap,
                                     EdgeConsts.TTL_FLOW_STR: 0}))

        g_new = networkx.DiGraph()
        g_new.add_nodes_from(list(nodes))
        g_new.add_edges_from(edges)

    return g_new


def transform_action(env, ac):
    if env.props.action_postprocess == ActionPostProcess.CLIP_ZERO:
        return np.clip(ac, HistoryConsts.EPSILON, None)
    elif env.props.action_postprocess == ActionPostProcess.SCALE_TO_ENV:
        return np.clip(env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low), env.action_space.low, env.action_space.high)
    elif env.props.action_postprocess == ActionPostProcess.SCALE_MIN:
        return ac-2*np.min(ac) if np.min(ac) < 0 else ac+2*np.min(ac)
    elif env.props.action_postprocess == ActionPostProcess.CEIL_TO_ENV:
        return np.ceil(
            np.clip(env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low), env.action_space.low, env.action_space.high)
            ) 