import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networking_env.environments.ecmp.env_args_parse import parse_args
from networking_env.environments.ecmp import history_env
from networking_env.environments.consts import SOMode
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
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
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
            #nn.Linear(128, 128),
            #nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

eps = 1e-16

def loss_fn(y_pred, y_true, env):
    num_nodes = env.get_num_nodes()
    opt = y_true[0][num_nodes * (num_nodes - 1)].item()
    y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

    commodities_to_paths = torch.from_numpy(env._optimizer._commodities_to_paths.A)
    paths_to_edges = torch.from_numpy(env._optimizer._paths_to_edges.A)
    
    y_pred = y_pred + eps
    paths_weight = torch.transpose(y_pred, 0, 1)
    commodity_total_weight = commodities_to_paths.matmul(paths_weight)
    commodity_total_weight = 1.0 / (commodity_total_weight)# + 1e-16)
    paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)
    paths_split = paths_weight.mul(paths_over_total)
    tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1))
    demand_on_paths = tmp_demand_on_paths.mul(paths_split)
    flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths)
    congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).transpose(0,1))
    max_cong = max(congestion)
    ret = 1.0 - max_cong if opt == 0.0 else max_cong/opt
    return ret

props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)

if props.so_mode == SOMode.TRAIN: #train
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #create the model
    model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths)
    model.double()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 1
    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            for (inputs, targets) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                yhat = model(inputs)
                loss = loss_fn(yhat, targets, env)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
                loss_sum += loss.item()
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
                test_loss = loss_fn(pred, targets, env)
                test_losses.append(test_loss.item())
            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
            #print statistics to file
            with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'so_stats.txt', 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses]
                dists.sort()
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
