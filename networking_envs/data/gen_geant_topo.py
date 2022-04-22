import os
import pathlib
import math
import urllib.request


topology_file = "http://informatique.umons.ac.be/networks/igen/downloads/geant-20041125.gml"
demand_matrices_dir = "directed-geant-uhlig-15min-over-4months-ALL-native"
output_dir = "GEANT"

train_fraction = 0.75 #fraction of DMs for the training dataset
num_of_dms_per_file = 2016
first_file_id = 1

#read nodes
#demand matrices
dms_input_files = os.listdir(demand_matrices_dir)
dms_input_files.sort()

nodes = {}
edges = {}
topology_name_to_dm_name = {}
with open(demand_matrices_dir + '/' + dms_input_files[0]) as f:
    line = f.readline().strip()
    while not line.startswith("NODES ("): line = f.readline().strip()
    line = f.readline().strip()
    while not line.startswith(")"):
        node_name = line.split()[0]
        nodes[node_name] = len(nodes)
        #topology_name = "SL" if node_name == 'si1.si' else node_name.split('.')[1].upper()
        topology_name = node_name.split('.')[0]
        topology_name_to_dm_name[topology_name] = node_name
        if topology_name == "de1": topology_name_to_dm_name["de2"] = node_name
        line = f.readline().strip()
        
topology_id_to_dm_id = {}
if topology_file.startswith('http'):
    f = urllib.request.urlopen(topology_file)
    lines = f.read().splitlines()
    f.close()
    lines = [line.decode('utf-8').strip() for line in lines]
else:
    f = open(topology_file)
    lines = f.read().splitlines()
    f.close()
    lines = [line.strip() for line in lines]

i = 0
while i < len(lines):
    if lines[i].startswith("node ["):
        node_id = None
        node_name = None
        while not lines[i].startswith("]"):
            i += 1
            if lines[i].startswith("id"):
                assert node_id == None
                node_id = lines[i].split()[1]
            elif lines[i].startswith("name") or lines[i].startswith("label"):
                assert node_name == None
                node_name = lines[i].split()[1][1:-1]
        assert node_id != None and node_name != None
        if node_name in topology_name_to_dm_name:
            assert node_id not in topology_id_to_dm_id
            topology_id_to_dm_id[node_id] = nodes[topology_name_to_dm_name[node_name]]
    elif lines[i].startswith("edge ["):
        edge_src = None
        edge_dst = None
        edge_cap = None
        while not lines[i].startswith("]"):
            i += 1
            if lines[i].startswith("source"):
                assert edge_src == None
                edge_src = lines[i].split()[1]
            elif lines[i].startswith("target"):
                assert edge_dst == None
                edge_dst = lines[i].split()[1]
            elif lines[i].startswith("LinkSpeedRaw") or lines[i].startswith("bandwidth"):
                assert edge_cap == None
                edge_cap = lines[i].split()[1]
        assert edge_src != None and edge_dst != None and edge_cap != None
        if edge_src in topology_id_to_dm_id and edge_dst in topology_id_to_dm_id:
            dm_src = int(topology_id_to_dm_id[edge_src])
            dm_dst = int(topology_id_to_dm_id[edge_dst])
            if dm_src != dm_dst:
                edges[(str(min(dm_src, dm_dst)), str(max(dm_src, dm_dst)))] = edge_cap

    i += 1

edges_list = [(int(e[0]), int(e[1]), edges[e]) for e in edges]
edges_list.sort()
    
assert not os.path.exists(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "/opts")
os.mkdir(output_dir + "/test")
os.mkdir(output_dir + "/train")

with open(output_dir + '/' + pathlib.PurePath(output_dir).name + "_int.pickle.nnet", "w", newline='\n') as f:
    for e in edges_list:
        f.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')

with open(output_dir + '/' + pathlib.PurePath(output_dir).name + "_int.txt", "w", newline='\n') as f:
    capacities = [["0.0"]*len(nodes) for _ in range(len(nodes))]
    for e in edges_list:
        capacities[e[0]][e[1]] = e[2]
        if (str(e[1]), str(e[0])) not in edges:
            capacities[e[1]][e[0]] = e[2]
    
    for i in range(len(nodes)):
        f.write(','.join(capacities[i]) + '\n')

#demand matrices
demands = [["0.0"]*(len(nodes)*len(nodes)) for _ in range(len(dms_input_files))]
count = 0
for i in range(len(dms_input_files)):
    file = dms_input_files[i]
    assert file.endswith('.txt')
    is_empty = True
    with open(demand_matrices_dir + '/' + file) as f:
        line = f.readline().strip()
        while not line.startswith("DEMANDS ("): line = f.readline().strip()
        line = f.readline().strip()
        while not line.startswith(")"):
            is_empty = False
            demand_info = line.split()
            src = demand_info[2]
            dst = demand_info[3]
            demand = demand_info[6]
            assert src in nodes and dst in nodes and demand_info[0] == src + '_' + dst and float(demand) >= 0.0
            demands[count][nodes[src]*len(nodes) + nodes[dst]] = repr(float(demand)*1e6)
            line = f.readline().strip()
    if is_empty:
        continue
    count += 1

#print("Number of demand matrices: " + str(count))

f_dm_idx = first_file_id
n_train_dms = int(math.ceil(count * train_fraction))
f = None
n_dms_in_f = 0
for i in range(count):
    if (n_dms_in_f % num_of_dms_per_file) == 0 or i == n_train_dms:
        if f != None: f.close()
        train_test = 'train' if i < n_train_dms else 'test'
        f = open(output_dir + '/' + train_test + '/' + str(f_dm_idx) + '.hist', 'w')
        f_dm_idx += 1
        n_dms_in_f = 0
        
    f.write(' '.join(demands[i]) + '\n')    
    n_dms_in_f += 1
