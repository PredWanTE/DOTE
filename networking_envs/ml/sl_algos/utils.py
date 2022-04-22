from data_gen import utils as DU
from networking_env.utils.shared_consts import SizeConsts
import numpy as np
import tqdm


def convert_to_batchs(X, Y, props):
    # shuffle order of things    
    xy_order = list( range(len(X)) )
    np.random.shuffle(xy_order)
    X = [X[_] for _ in xy_order]
    Y = [Y[_] for _ in xy_order]
    
    # now convert X,Y into batches of size props.sl_batch_size
    X_ = []
    Y_ = []
    for i in range(0, len(X), props.sl_batch_size):
        if i+props.sl_batch_size< len(X):
            X_.append( np.stack( X[i:i+props.sl_batch_size] ) )
            Y_.append( np.stack( Y[i:i+props.sl_batch_size] ) )
    else:
        # add the final batch if we have anything left
        if i == len(X)-1:
            X_.append( np.stack( X[i:] ) )
            Y_.append( np.stack( Y[i:] ) )  
    return np.asarray(X_), np.asarray(Y_)


def basic_through_time(tms, props):
    tms_per_hour = 60 // props.tm_sample_rate_min
    tms_per_day = 24*tms_per_hour

    num_tms = len(tms)
    
    X_prevtimes = []
    X_prev_through_time = []
    Y_ = []
    tm_start_id = max(tms_per_day * props.look_len, props.hist_len)
    tm_end_id = num_tms 
    for now_predicting_ind in range(tm_start_id, tm_end_id):
        Y_.append( tms[ now_predicting_ind ] )
        X_prevtimes.append( np.vstack(
                                tms[now_predicting_ind-props.hist_len:now_predicting_ind] 
                            ))
        X_prev_through_time.append( np.vstack( [tms[ now_predicting_ind - tms_per_day*i] 
                                                for i in range(props.look_len,0,-1)] ) )
        
    X_ = [np.asarray(X_prevtimes), np.asarray(X_prev_through_time)]
            
    return X_, np.vstack(Y_)


def basic_per_commodity(tms, props, y_data, num_nodes):
    X_dict = {}
    Y_dict = {}
    # tms are assumed to be a single vector of size N*N x 1
    np_tms = np.vstack(tms)
    np_tms = np_tms.T

    run_ar = False
    if run_ar:
        np_tms_flat = np.delete(np_tms, [idx + num_nodes * idx for idx in range(num_nodes)], axis=0)
        np_tms_flat = np_tms_flat.flatten('F')

    tm_nodes = np_tms[2:]
    tm_times = np_tms[:2]

    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst: continue
            
            commodity_id = src * num_nodes + dst
            
            X_ = []
            Y_ = []
        
            # we map from k-long previous steps into the next demand
            for histid in range(len(tms) - props.hist_len):

                if props.time is True:
                    x_values = tm_nodes[commodity_id, histid:histid + props.hist_len].copy()
                    time_of_x = tm_times[:, histid:histid + props.hist_len]
                    x_series = np.hstack([time_of_x[0, :], time_of_x[1, :], x_values])
                    X_.append(x_series)
                    if y_data is None:
                        Y_.append(tm_nodes[commodity_id, histid + props.hist_len])
                    else:
                        Y_.append(y_data[commodity_id])

                else:
                    if run_ar:
                        start_idx = histid*num_nodes*(num_nodes-1)
                        end_idx = start_idx + props.hist_len*num_nodes*(num_nodes-1)
                        X_.append(np_tms_flat[start_idx:end_idx])
                    else:
                        if props.hist_len == 0:
                            X_.append(np.zeros(np_tms[commodity_id, 0:1].shape))
                        else:
                            X_.append(np_tms[commodity_id, histid:histid + props.hist_len].copy())
                    if y_data is None:
                        Y_.append(np_tms[commodity_id, histid + props.hist_len])
                    else:
                        Y_.append(y_data[commodity_id])

            assert len(X_) == len(Y_)
            X_dict[(src, dst)] = X_
            Y_dict[(src, dst)] = Y_
    return X_dict, Y_dict 


def basic_stack(tms, props, y_data=None):
    X_ = []
    Y_ = []

    np_tms = np.vstack(tms)

    # we map from k-long histories into the next DM
    for histid in tqdm.tqdm(range(len(tms) - props.hist_len)):
        if props.time is True:
            X_.append(np.hstack([
                np.vstack(np_tms[histid:histid + props.hist_len, :2]),
                np.vstack(np_tms[histid:histid + props.hist_len, 2:])
            ]))
        else:
            X_.append(np.vstack(np_tms[histid:histid + props.hist_len]))
        if y_data is None:
            Y_.append(np_tms[histid + props.hist_len])
        else:
            Y_.append(y_data[histid + props.hist_len])
    assert len(X_) == len(Y_)
    
    return np.asarray(X_), np.vstack(Y_)  # convert_to_batches(X_, Y_, props)


def basic_stack_ae(tms, props, y_date=None):
    X_ = np.vstack(tms)[:, 2:]

    return np.asarray(X_), None  # convert_to_batches(X_, Y_, props)


def basic_scale_per_comm(train, test, skip=True):
    if isinstance(train, list):
        train = np.stack(train)
        test = np.stack(test)

    test_org = test.copy()

    if not skip:
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        # normalize data
        std[std == 0] = 1
        train = (train - mean) / std
        test = (test - mean) / std

        # this happens when we have the diagonal of the TM
        train[np.isnan(train)] = 0
        test[np.isnan(test)] = 0
    else:
        std = np.ones(shape=train.std(axis=0).shape)
        mean = np.zeros(shape=train.mean(axis=0).shape)
        act_mean = train.mean(axis=0)

    return np.vsplit(train, train.shape[0]), \
           np.vsplit(test, test.shape[0]), \
           {'std': std,
            'mean': mean,
            'act_mean': act_mean,
            "test_org": test_org}


# now convert these into numpy arrays
def get_data(hists, demand_scale=None, sep=" "):
    # SizeConsts.BPS_TO_MBPS
    tms = []
    for h in tqdm.tqdm(hists):
        lines = open(h).readlines()
        hist_tms = [np.array([float(_) for _ in line.split(sep) if _], dtype=np.float32) for line in lines]
        tms += hist_tms
    if demand_scale:
        tms = [demand_scale(tm) for tm in tms]
    return tms


def get_train_test_tms(props):
    train_hist_files, test_hist_files = DU.get_train_test_files(props)

    train_tms = get_data(train_hist_files)
    test_tms = get_data(test_hist_files)
    
    return train_tms, test_tms


def get_train_test(props, make_XY=basic_stack, preprocess_data=basic_scale_per_comm):
    train_tms, test_tms = get_train_test_tms(props)
    train_tms, test_tms, extra_data = preprocess_data(train_tms, test_tms)
    y_data_train = None
    y_data_test = None
    
    if props.use_imitation:
        train_hist_files, test_hist_files = DU.get_train_test_files(props)
        train_tunnels = [f + ".tunnels" for f in train_hist_files]
        test_tunnels = [f + ".tunnels" for f in test_hist_files]
        y_data_train = get_data(train_tunnels, sep=",", demand_scale=None)
        y_data_test = get_data(test_tunnels, sep=",", demand_scale=None)

    train = make_XY(train_tms, props, y_data_train)
    test = make_XY(test_tms, props, y_data_test)
    
    if props.compute_opts and props.compute_opts_dir == "train":
        test = train
        test_tms = train_tms

    return (train, train_tms), (test, test_tms), extra_data


def train_eval(model, X_train, Y_train, X_test, Y_test, props, **args):
    fit_res = model.fit(X_train, Y_train, **args)
    try:
        Y_hat = model.predict(X_test)
    except:
        import pdb; pdb.set_trace()
#     score = model.score(X_test, Y_test)
    return Y_hat, fit_res


def dump_model(props, model, fname):
    import joblib

    joblib.dump(model, fname) 


def dump_prediciton(props, y_hat, y, mse=0, dir_name=None):
    import joblib
    import os
    logs_dir = dir_name or props.dir_name + "/" + props.sl_model_type
    os.makedirs(logs_dir, exist_ok=True)
    joblib.dump(y_hat, '%s/yhat.pkl' % logs_dir)
    joblib.dump(y, '%s/y.pkl' % logs_dir)


def get_all_cplex_res(props, y_hat, y):
    from tqdm import trange
    assert y_hat.shape == y.shape
    tr = trange(y_hat.shape[0], leave=True)
    ress = []
    ress_all = []
    cntr = 0
    for i in tr:
        tm = y_hat[i, :]
        next_tm = y[i, :]
        factor_units = 1.0 #SizeConsts.ONE_Mb
        res_str = DU.get_opt_cplex(props, tm * factor_units, next_tm * factor_units, opt_function=props.opt_function,
                                   use_cplex=False, idx=cntr, path=props.base_path+'/data/'+props.ecmp_topo+'/' + props.opts_dir + '/')
        cntr += 1
        try:
            ress.append(float(res_str[-1].split(":")[1]))
        except:
            import pdb; pdb.set_trace()
        ress_all.append(res_str)
    return ress, ress_all


def compute_cplex_res(props):
    logs_dir = props.dir_name + "/" + props.sl_model_type
    
    # first load the model files
    import joblib
    y_hat = joblib.load('%s/yhat.pkl' % logs_dir)
    y = joblib.load('%s/y.pkl' % logs_dir)

    ress, ress_all = get_all_cplex_res(props, y_hat, y)
    import os
    os.makedirs(logs_dir, exist_ok=True)
    fname = logs_dir + "/cplex_res.out"
    with open(fname, 'w') as f:
        f.write("AVG: %f, MEDIAN: %f, 90th: %f, 95th: %f, 99th: %f" %
                (np.average(ress),
                 np.median(ress),
                 np.percentile(ress, 90),
                 np.percentile(ress, 95),
                 np.percentile(ress, 99)))
    joblib.dump(ress_all, '%s/all_cplex_res.pkl' % logs_dir)
