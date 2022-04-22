#Convert GML format to DOTE format

import os
import numpy as np
from numpy import random

src_dir = "zoo_topologies"
network_name = "Abilene"
dest_dir = network_name

#additional configuration variables
default_capacity = "10000.0"
demands_factor = 10.0
n_train_matrices = 3
n_test_martices = 1

input_file_name = src_dir + '/' + network_name + ".gml"

nodes = set()
edges = {}
assert os.path.exists(input_file_name)
with open(input_file_name) as f:
    while True:
        line = f.readline().strip()
        if not line: break
        if line.startswith("node ["): #read node info
            node_id = None
            while not line.startswith("]"):
                if line.startswith("id "):
                    assert node_id == None #single id per node
                    node_id = line[len("id "):]
                line = f.readline().strip()

            assert node_id != None and node_id not in nodes
            nodes.add(node_id)
            continue

        elif line.startswith("edge ["):
            edge_src = None
            edge_dst = None
            capacity = default_capacity
            while not line.startswith("]"):
                if line.startswith("source "):
                    assert edge_src == None
                    edge_src = line[len("source "):]
                elif line.startswith("target "):
                    assert edge_dst == None
                    edge_dst = line[len("target "):]
                elif line.startswith("LinkSpeedRaw "):
                    capacity = line[len("LinkSpeedRaw "):]
                line = f.readline().strip()

            assert edge_src != None and edge_dst != None
            edges[(edge_src, edge_dst)] = capacity
            continue
        
#verification:
#1. nodes are numbered 0 to len(nodes)
#2. edges src and targets are existing nodes
for i in range(len(nodes)): assert(str(i)) in nodes
for e in edges: assert e[0] in nodes and e[1] in nodes

#Convert to DOTE format
assert not os.path.exists(dest_dir)
os.mkdir(dest_dir)
os.mkdir(dest_dir + "/opts")
os.mkdir(dest_dir + "/test")
os.mkdir(dest_dir + "/train")

edges_list = [(int(e[0]), int(e[1]), edges[e]) for e in edges]
edges_list.sort()

with open(dest_dir + '/' + dest_dir + "_int.pickle.nnet", "w", newline='\n') as f:
    for e in edges_list:
        f.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')

with open(dest_dir + '/' + dest_dir + "_int.txt", "w", newline='\n') as f:
    capacities = [["0.0"]*len(nodes) for x in range(len(nodes))]
    for e in edges_list:
        capacities[e[0]][e[1]] = e[2]
        if (str(e[1]), str(e[0])) not in edges:
            capacities[e[1]][e[0]] = e[2]
    
    for i in range(len(nodes)):
        f.write(','.join(capacities[i]) + '\n')

#generate random traffic matrices
node_to_n_edges = {}
total_edges_cap = 0.0
for n in nodes: node_to_n_edges[n] = 0
for e in edges:
    node_to_n_edges[e[0]] += 1
    total_edges_cap += float(edges[e])
    if (e[1], e[0]) not in edges:
        node_to_n_edges[e[1]] += 1
        total_edges_cap += float(edges[e])
    
    
total_edges = sum(node_to_n_edges.values())

#print("#nodes = {0}, #edges = {1}, total capacity {2}".format(str(len(nodes)), str(len(edges)), str(total_edges_cap)))

total_demands = total_edges_cap / demands_factor
frac_dict = {}
for u in range(len(nodes)):
    for v in range(len(nodes)):
        if u == v:
            frac_dict[(u, v)] = 0.0
        else:
            u_str = str(u)
            v_str = str(v)
            frac_dict[(u, v)] = (node_to_n_edges[u_str] * node_to_n_edges[v_str]) / (total_edges * (total_edges - node_to_n_edges[u_str]))
            
#gen train and test DMs
for m_idx in range(1, n_train_matrices + n_test_martices + 1):
    if m_idx <= n_train_matrices: fname = dest_dir + '/train/' + str(m_idx) + '.hist'
    else: fname = dest_dir + '/test/' + str(m_idx) + '.hist'
    f = open(fname, 'w')
    for dm_idx in range(2016):
        demands = ["0.0"]*(len(nodes)*len(nodes))
        for u in range(len(nodes)):
            for v in range(len(nodes)):
                if u == v: continue
                frac = frac_dict[(u,v)]
                # sample from gaussian with mean = frac, stddev = frac / 4
                demands[u*len(nodes) + v] = f"{(total_demands * max(np.random.normal(frac, frac / 4), 0.0)):.6g}"
        f.write(' '.join(demands) + '\n')
    f.close()
