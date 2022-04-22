from ml.ae_algos import archs
from ml.sl_algos import utils as SLU
from ml.sl_algos.nn import utils as NNU

import numpy as np


def train_eval(model,
               X_train, Y_train, 
               X_test, Y_test, 
               props):
    
    NNU.compile_model(model, props)

    # print the model summary before fitting
    print(model.summary())

    Y_hat, history = SLU.train_eval(model, 
                       X_train, Y_train, 
                       X_test, Y_test, 
                       props, **NNU.get_extra_nn(props, X_test, Y_test))

    # dump prediction
    SLU.dump_prediciton(props, Y_hat, Y_test)

    
    # dump model to file
    NNU.dump_model(props, model) 

    
    # compute cplex things
    SLU.compute_cplex_res(props)

def main(env, props, train, test):
    X_train, Y_train = train
    X_test,Y_test = test
    model = archs.get_model(env,props)
    train_eval(model, 
               X_train, Y_train, 
               X_test, Y_test, 
               props)


