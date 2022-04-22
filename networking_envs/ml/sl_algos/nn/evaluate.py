from ml.sl_algos.nn import archs
from ml.sl_algos import utils as SLU
from ml.sl_algos.nn import utils as NNU

import numpy as np

def train_eval(model,
               X_train, Y_train, 
               X_test, Y_test, 
               props, extra_data):
    
    NNU.compile_model(model, props)

    print(model.summary())

    Y_hat, history = SLU.train_eval(model, 
                       X_train, Y_train,
                       X_test, Y_test, 
                       props, **NNU.get_extra_nn(props, X_test, Y_test))

    # dump prediction
    Y_hat = (Y_hat*extra_data['std']) + extra_data['mean']
    Y_test = (Y_test*extra_data['std']) + extra_data['mean']
    
    SLU.dump_prediciton(props, Y_hat, Y_test, NNU.get_dir(props))

    # dump model to file
    NNU.dump_model(props, model)


def main(env, props, train, test, extra_data):
    X_train, Y_train = train
    X_test,Y_test = test
    model = archs.get_model(env, props)
    train_eval(model, 
               X_train, Y_train, 
               X_test, Y_test, 
               props, extra_data)