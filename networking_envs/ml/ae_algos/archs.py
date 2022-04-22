from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def get_basic_ae(env,props):
    num_nodes = env._num_nodes
    
    x_input = Input(shape=(num_nodes*num_nodes, 1))
    
    for ind,params in enumerate(props.sl_model_params.split(",")):
        params = params.split(":")
        units = int(params[0]) 
        activation = params[1]
        if ind == 0:
            encoded = Dense(units, activation=activation)(x_input)
        else:
            encoded = Dense(units, activation=activation)(encoded)
    
    for ind,params in enumerate(props.sl_model_params.split(",")[::-1]):
        params = params.split(":")
        units = int(params[0]) 
        activation = params[1]
        if ind == 0:
            decoded = Dense(units, activation=activation)(encoded)
        else:
            decoded = Dense(units, activation=activation)(decoded)
    
    
    encoder = Model(x_input, encoded)
    decoder = Model(encoded, decoded)
    
    model = Sequential()
    model.add(Flatten(input_shape=(1,num_nodes*num_nodes)))
    model.add(Dense())

def get_model(env, props):
    model = get_basic_ae(env,props)
    
    return model


