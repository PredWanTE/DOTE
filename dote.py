import sys
import os

cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networking_env.environments.ecmp.env_args_parse import parse_args
from networking_env.environments.ecmp import history_env
from networking_env.environments.consts import SOMode
from networking_env.utils.shared_consts import SizeConsts
from tqdm import tqdm

# dataset definition
class DmDataset(Dataset):
    def __init__(self, props=None, env=None, is_test=None):
        # store the inputs and outputs
        assert props != None and env != None and is_test != None

        num_nodes = env.get_num_nodes()
        env.test(is_test)
        tms = env._simulator._cur_hist._tms
        opts = env._simulator._cur_hist._opts
        tms = [np.asarray([tms[i]]) for i in range(len(tms))]
        np_tms = np.vstack(tms)
        np_tms = np_tms.T
        np_tms_flat = np_tms.flatten('F')

        assert (len(tms) == len(opts))
        X_ = []
        for histid in range(len(tms) - props.hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)
            end_idx = start_idx + props.hist_len * num_nodes * (num_nodes - 1)
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_)
        self.y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(opts))])


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

# model definition
class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxUtil, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

def loss_fn_maxutil(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()
    
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

    
        y_pred = y_pred + 1e-16 #eps
        paths_weight = torch.transpose(y_pred, 0, 1)
        commodity_total_weight = commodities_to_paths.matmul(paths_weight)
        commodity_total_weight = 1.0 / (commodity_total_weight)
        paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
        paths_split = paths_weight.mul(paths_over_total)
        tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1))
        demand_on_paths = tmp_demand_on_paths.mul(paths_split)
        flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths)
        congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).transpose(0,1))
        max_cong = torch.max(congestion)
        
        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item()
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt
        losses.append(loss)
        loss_vals.append(loss_val)
    
    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val

class NeuralNetworkMaxFlowMaxConc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxFlowMaxConc, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_dim),
            nn.ELU(alpha=0.1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

def loss_fn_maxflow_maxconc(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()

    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

        y_pred = y_pred + 0.1 #ELU
        edges_weight = paths_to_edges.transpose(0,1).matmul(torch.transpose(y_pred, 0, 1))
        alpha = torch.max(edges_weight.divide(torch.tensor(np.array([env._capacities])).transpose(0,1)))
        max_flow_on_tunnel = y_pred / alpha
        max_flow_per_commodity = commodities_to_paths.matmul(max_flow_on_tunnel.transpose(0,1))

        if props.opt_function == "MAXFLOW": #MAX FLOW
            max_mcf = torch.sum(torch.minimum(max_flow_per_commodity.transpose(0,1), y_true))
            
            loss = -max_mcf if max_mcf.item() == 0.0 else -max_mcf/max_mcf.item()
            loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
        
        elif props.opt_function == "MAXCONC": #MAX CONCURRENT FLOW
            actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            max_concurrent_vec = torch.full_like(actual_flow_per_commodity, fill_value=1.0)
            mask = y_true != 0
            max_concurrent_vec[mask] = actual_flow_per_commodity[mask].divide(y_true[mask])
            max_concurrent = torch.min(max_concurrent_vec)
            
            #actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            #actual_flow_per_commodity = torch.maximum(actual_flow_per_commodity, torch.tensor([1e-32]))
            #max_concurrent = torch.min(actual_flow_per_commodity.divide(torch.maximum(y_true, torch.tensor([1e-32])))
            
            loss = -max_concurrent if max_concurrent.item() == 0.0 else -max_concurrent/max_concurrent.item()
            loss_val = 1.0 if opt == 0.0 else max_concurrent.item()/opt
                
            #update concurrent flow statistics
            if concurrent_flow_cdf != None:
                curr_dm_conc_flow_cdf = [0]*len(concurrent_flow_cdf)
                for j in range(env.get_num_nodes() * (env.get_num_nodes() - 1)):
                    allocated = max_flow_per_commodity[j][0].item()
                    actual = y_true[0][j].item()
                    curr_dm_conc_flow_cdf[j] = 1.0 if actual == 0 else min(1.0, allocated / actual)
                curr_dm_conc_flow_cdf.sort()
                
                for j in range(len(curr_dm_conc_flow_cdf)):
                    concurrent_flow_cdf[j] += curr_dm_conc_flow_cdf[j]
        else:
            assert False
        
        losses.append(loss)
        loss_vals.append(loss_val)

    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val
    
    
props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)

ctp_coo = env._optimizer._commodities_to_paths.tocoo()
commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), torch.DoubleTensor(ctp_coo.data), torch.Size(ctp_coo.shape))
pte_coo = env._optimizer._paths_to_edges.tocoo()
paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), torch.DoubleTensor(pte_coo.data), torch.Size(pte_coo.shape))

batch_size = props.so_batch_size
n_epochs = props.so_epochs
concurrent_flow_cdf = None
if props.opt_function == "MAXUTIL":
    NeuralNetwork = NeuralNetworkMaxUtil
    loss_fn = loss_fn_maxutil
elif props.opt_function == "MAXFLOW":
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
elif props.opt_function == "MAXCONC":
    if batch_size == 1:
        batch_size = props.so_max_conc_batch_size
        n_epochs = n_epochs*batch_size
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
    if props.so_mode == SOMode.TEST:
        concurrent_flow_cdf = [0] * (env.get_num_nodes()*(env.get_num_nodes()-1))
else:
    print("Unsupported optimization function. Supported functions: MAXUTIL, MAXFLOW, MAXCOLC")
    assert false

if props.so_mode == SOMode.TRAIN: #train
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #create the model
    model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths)
    model.double()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            for (inputs, targets) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                yhat = model(inputs)
                loss, loss_val = loss_fn(yhat, targets, env)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss_val)
                loss_sum += loss_val
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss=loss_avg)

    #save the model
    torch.save(model, 'model_dote.pkl')

elif props.so_mode == SOMode.TEST: #test
    # create the dataset
    test_dataset = DmDataset(props, env, True)
    # create a data loader for the test set
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #load the model
    model = torch.load('model_dote.pkl')
    model.eval()
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses = []
            for (inputs, targets) in tests:
                pred = model(inputs)
                test_loss, test_loss_val = loss_fn(pred, targets, env)
                test_losses.append(test_loss_val)
            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
            #print statistics to file
            with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'so_stats.txt', 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses]
                dists.sort(reverse=False if props.opt_function == "MAXUTIL" else True)
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
            
            if concurrent_flow_cdf != None:
                concurrent_flow_cdf.sort()
                with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'concurrent_flow_cdf.txt', 'w') as f:
                    for v in concurrent_flow_cdf:
                        f.write(str(v / len(dists)) + '\n')
