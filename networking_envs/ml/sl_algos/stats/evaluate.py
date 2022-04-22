from ml.sl_algos.nn import utils as NNU
from sklearn.metrics import mean_squared_error as MSE
import joblib
import numpy as np
from tqdm import tqdm, trange

from ml.sl_algos import utils as SLU
from networking_env.utils.shared_consts import SizeConsts
from data_gen import utils as datagen_utils


def loss_transform(Y_test,
                   Y_hat,
                   extra_data,
                   commid,
                   loss=MSE):
    try:
        if isinstance(Y_test, list):
            Y_test = np.array(Y_test)
        Y_test = Y_test * extra_data['std'][commid] + extra_data['mean'][commid]
        Y_hat = Y_hat * extra_data['std'][commid] + extra_data['mean'][commid]
        return Y_test, Y_hat, np.sqrt(MSE(Y_test, Y_hat))
    except:
        import pdb; pdb.set_trace()


def main_model(model_name,
               pairs,
               all_train_X,
               all_train_Y,
               all_test_X,
               all_test_Y,
               env,
               props,
               extra_data):
    res = {}
    models_res = {}
    
    is_nn = "nn" in props.sl_type
    
    # set models module (nn/stats)
    # we assume we have train/test per commodity
    # lets train a model for each and every one of these commodities
    for src, dst in tqdm(pairs):
        if src == dst:
            continue
        #if src != 1: continue
        #if dst != 9: continue
#         
#         from statsmodels.tsa.x13 import x13_arima_select_order as X13
#         import pandas as pd
#         import pdb;pdb.set_trace()
#         res = X13(pd.DataFrame(aa[src*11+dst]),x12path="/Applications/x12arima", )
#         
        
        X_train, Y_train = all_train_X[(src, dst)], all_train_Y[(src, dst)]
        X_test, Y_test = all_test_X[(src, dst)], all_test_Y[(src, dst)]
        models_ = None
        if is_nn:
            from ml.sl_algos.nn import archs as models
            extra_model_data = NNU.get_extra_nn(props, np.vstack(X_test), np.vstack(Y_test))
            models_ = models
            
        else:
            from ml.sl_algos.stats import models
            extra_model_data = {}
            models_ = models

        model = models_.get_model(env, props, model_name)
        if is_nn: 
            NNU.compile_model(model, props)

        if is_nn:
            Y_hat, _ = SLU.train_eval(model,
                                      X_train,
                                      Y_train,
                                      X_test,
                                      Y_test,
                                      props,
                                      **extra_model_data)
            Y_hat = Y_hat.reshape((-1,))
        else:
            Y_hat, _ = SLU.train_eval(model,
                                      X_train,
                                      Y_train,
                                      X_test,
                                      Y_test,
                                      props,
                                      **extra_model_data)
        
        # test, hat, loss
        res[(src, dst)] = loss_transform(Y_test, Y_hat, extra_data, src * env.get_num_nodes() + dst)
        models_res[(src, dst)] = model
        
    tm_combined_hat = []
    tm_combined_real = []

    num_tm_test = len(all_test_Y[(0, 1)])
    
    # now that we have all results, we can reconstruct the matrices
    for src, dst in tqdm(pairs):
        # add zero column when using commodity (i,i)
        if src == dst:
            tm_combined_real.append(np.zeros((num_tm_test,)))
            tm_combined_hat.append(np.zeros((num_tm_test,)))
        else:
            tm_combined_real.append(res[(src, dst)][0])
            tm_combined_hat.append(res[(src, dst)][1])
    
    tm_combined_hat = [np.reshape(_, (-1, 1)) for _ in tm_combined_hat]
    tm_combined_real = [np.reshape(_, (-1, 1)) for _ in tm_combined_real]
    tm_combined_hat_ = np.concatenate(tm_combined_hat, axis=1)
    tm_combined_real_ = np.concatenate(tm_combined_real, axis=1)
    
    tm_combined_hat_[tm_combined_hat_ < 0] = 0
    tm_combined_real_[tm_combined_real_ < 0] = 0

    # get predictions for test matrices
    mse = np.power(tm_combined_real_ - tm_combined_hat_, 2)

    # calculate MSE for each commodity
    mse = np.sum(mse, axis=0) / mse.shape[0]
    
    return models_res, tm_combined_hat_, tm_combined_real_, np.sqrt(mse)


def main(env, props, train, test, extra_data):
    all_train_X, all_train_Y = train[0]
    all_test_X, all_test_Y = test[0]
    
    is_nn = "nn" in props.sl_type
    
    pairs = []
#     if is_nn:
#         assert props.sl_src > -1
#         assert props.sl_dst > -1
#         assert props.sl_dst != props.sl_src
#         pairs.append((props.sl_src, props.sl_dst))
#     else:
    for src in range(env.get_num_nodes()):
        for dst in range(env.get_num_nodes()):
            pairs.append((src, dst))
    
    results = []
    model_res = main_model(props.sl_model_type,
                           pairs,
                           all_train_X,
                           all_train_Y,
                           all_test_X,
                           all_test_Y,
                           env,
                           props,
                           extra_data)
    results.append(model_res)
    import os
    logs_dir = props.dir_name + "/" + props.sl_model_type
    os.makedirs(logs_dir, exist_ok=True)
    mses = np.array([0.])
    print("\n[+] Finished training.")
    if props.no_dump is False:
        for pair in tqdm(pairs):
            if pair[0] == pair[1]:
                continue
            src = pair[0]
            dst = pair[1]
            fname = '%s/model_src_%d_dst_%d'%(logs_dir, src, dst)
            if is_nn:
                fname = '%s/model_src_%d_dst_%d'%(NNU.get_dir(props), src, dst)
                NNU.dump_model( props, model_res[0][pair], fname)
                SLU.dump_prediciton(props, model_res[1], model_res[2], NNU.get_dir(props))
            else:
                SLU.dump_model(props, model_res[0][pair], fname)
        mses = np.concatenate((mses, model_res[-1]))
        SLU.dump_prediciton(props, model_res[1], model_res[2])
    from matplotlib import pyplot as plt
    # plt.plot(sorted(mses))
    # mses_f = mses.flatten()

    actual_mean = extra_data['act_mean']
    # # uncomment this for MSEs
    # from plotting.utils import get_cdf
    #
    # # plot CDF of RMSE divided by mean
    mses = np.delete(mses, [0], axis=0)

    #joblib.dump(actual_mean,'C:\\Users\\user\\Desktop\\means.pkl')
    #joblib.dump(mses, 'C:\\Users\\user\\Desktop\\mses.pkl')

    # plt.plot(*get_cdf(actual_mean))
    # plt.show()
    #
    # zero_means = np.where(actual_mean == 0)[0]
    # actual_mean[zero_means] = 1
    #
    # plt.plot(*get_cdf(mses / actual_mean))
    # plt.show()
    # plt.plot(*get_cdf(mses))
    # plt.show()


#         get_all_cplex_res(props, model_res[1], model_res[2])


# def main1(env, props, train, test, extra_data):
#     all_train_X, all_train_Y = train[0]
#     all_test_X, all_test_Y = test[0]
#     
#     all_train_tms = train[1]
#     all_test_tms = test[1]
#     
#     res = {}
#     
#     pairs = []
#     for src in range(env.get_num_nodes()):
#         for dst in range(env.get_num_nodes()):
#             pairs.append((src,dst))
#     
#     # we assume we have train/test per commoditiy
#     # lets train a model for each and every one of these commodities
#     for src,dst in tqdm(pairs) :
#         if src == dst: continue
#         X_train, Y_train = all_train_X[(src, dst)], all_train_Y[(src,dst)]
#         X_test, Y_test = all_test_X[(src, dst)], all_test_Y[(src,dst)]
#         model = models.get_model(env,props)
#         Y_hat, _ = SLU.train_eval(model, 
#                        X_train, Y_train, 
#                        X_test, Y_test, 
#                        props)
#         # test, hat, loss
#         res[(src,dst)] = loss_transform(Y_test, Y_hat, extra_data, src*env.get_num_nodes()+dst)
#     
#     
#     tm_combined_hat = []
#     tm_combined_real = []
# 
#     num_tm_test = len(all_test_Y[(0,1)])
#     
#     # now that we have all results, we can reconstruct the matrices
#     for src,dst in tqdm(pairs) :
#         # add zero column when using commodity (i,i)
#         if src == dst:
#             tm_combined_real.append(np.zeros((num_tm_test,)))
#             tm_combined_hat.append(np.zeros((num_tm_test,)))
#         else:
#             tm_combined_real.append(res[(src,dst)][0])
#             tm_combined_hat.append(res[(src,dst)][1])
#     
#     tm_combined_hat = [np.reshape(_, (-1,1)) for _ in tm_combined_hat]
#     tm_combined_real = [np.reshape(_, (-1,1)) for _ in tm_combined_real]
#     tm_combined_hat_ = np.concatenate(tm_combined_hat, axis=1)
#     tm_combined_real_ = np.concatenate(tm_combined_real, axis=1)
#     
#     tm_combined_hat_[tm_combined_hat_<0] = 0
#     tm_combined_real_[tm_combined_real_<0] = 0
#     mse = np.power(tm_combined_real_ - tm_combined_hat_,2)
#     mse = np.sum(mse,axis=1)/mse.shape[1]
#     
#     ress = []
#     tr = trange(100, leave=True) #tm_combined_hat_.shape[0]
#     for i in tr:
#         if len(ress):
#             tr.set_description("AVG: %f, MEDIAN: %f, 90th: %f, 95th: %f, 99th: %f"%( np.average(ress), np.median(ress), np.percentile(ress,90), np.percentile(ress,95), np.percentile(ress,99)))
#         tm = tm_combined_hat_[i,:]
#         next_tm = tm_combined_real_[i,:]
#         
#         res_str = datagen_utils.get_opt_cplex(props, tm*SizeConsts.ONE_Mb, next_tm*SizeConsts.ONE_Mb)
#         ress.append( float(res_str[-1].split(":")[1]) )
#     
# #     from matplotlib import pyplot as plt
# #     mse_t = mse[mse<2]
# #     plt.plot(sorted(mse_t),np.linspace(0, len(mse_t)/len(mse), len(mse_t)))
# #     plt.show()
#     import pdb;pdb.set_trace()
