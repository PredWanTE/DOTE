from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input, concatenate, Convolution1D, SeparableConv2D, Reshape, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM as LSTM_NN
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import MaxPooling1D, AvgPool1D, MaxPool1D

FCN = "fcn"
FCN1 = "fcn1"

MULTI_FCN = "multi_fcn"
MULTI_1D_CNN = "multi_1d_cnn"
MULTI_2D_CNN = "multi_2d_cnn"

CNN_1D_COMM = "cnn1d_comm"
CNN_1D_DEMAND = "cnn1d_demand"
CNN_2D_DEMAND = "cnn2d_demand"

LSTM_DM = "lstm_dm"
LSTM_IMITATION = "lstm_im"
BIDER_LSTM_DM = "bider_lstm_dm"
BIDER_LSTM_IMITATION = "bider_lstm_im"
 
def fcn(env, props):
    num_nodes = env._num_nodes
    num_edges = env._num_edges

    time = 0
    if props.time is True:
        time = 2
    model = Sequential(
            (Flatten(input_shape=(props.hist_len, num_nodes * num_nodes + time)),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(num_nodes * num_nodes + time)
            )
        )
      
    return model


def fcn1(env, props):
    model = Sequential(
            (Dense(props.hist_len*2, activation='elu', input_shape=(props.hist_len, )),
            Dense(props.hist_len*2, activation='elu'),
            Dense(props.hist_len, activation='elu'),
            Dense(1))
        )

    return model

def cnn1d_comm(env, props):
    model = Sequential((
        Reshape((props.hist_len,1), input_shape=(props.hist_len,) ),
        Convolution1D(filters=props.hist_len*2, 
                      kernel_size=4, 
                      activation='tanh', 
                ),
        Convolution1D(filters=props.hist_len, 
                      kernel_size=4, 
                      activation='tanh'
                ),
        Flatten(),
        Dense(props.hist_len, activation='elu'),
        Dense(1)
    ))

    return model

def cnn1d_demand(env, props):
    num_nodes = env._num_nodes
    model = Sequential((
        Convolution1D(filters=props.hist_len*2, 
                      kernel_size=4, 
                      activation='tanh', 
                      input_shape=(props.hist_len,num_nodes**2)
                ),
        Convolution1D(filters=props.hist_len, 
                      kernel_size=4, 
                      activation='tanh'
                ),
        Flatten(),
        Dense(num_nodes**2, activation='elu'),
        Dense(num_nodes**2)
    ))

    return model


def cnn2d_demand(env, props):
    num_nodes = env._num_nodes
    model = Sequential((
        Reshape((num_nodes, num_nodes, props.hist_len),
                input_shape=(num_nodes**2, props.hist_len)
                ),
        Conv2D(filters=256, #props.hist_len*2,
                        kernel_size=(1, 1),
                        activation='relu',
                        input_shape=(props.hist_len, num_nodes**2)
                        ),
        Conv2D(filters=128,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=64,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=32,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=16,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=8,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=4,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=2,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Conv2D(filters=1,  # props.hist_len*2,
               kernel_size=(1, 1),
               activation='relu',
               input_shape=(props.hist_len, num_nodes ** 2)
               ),
        Flatten()
    ))

    return model


def get_lstm_layer(units, return_sequences=False, input_shape=None, bidirection=False):
        if bidirection:
            lstm = LSTM_NN(units, return_sequences=return_sequences)
            return Bidirectional(lstm, input_shape=input_shape) \
                    if input_shape else Bidirectional(lstm)
        else:
            return LSTM_NN(units, return_sequences=return_sequences, input_shape=input_shape) \
                    if input_shape else LSTM_NN(units, return_sequences=return_sequences)


def lstm(env, props, demands=True, bidirection=False):
    model = Sequential()
    num_nodes = env._num_nodes
    
    # iterate params to create rest of architecture
    for ind, params in enumerate(props.sl_model_params.split(",")):
        params = params.split(":")
        units = int(params[0]) 
        activation = params[1]
        if ind == 0:
            model.add( get_lstm_layer(units, 
                                      return_sequences=True, 
                                      input_shape=(props.hist_len, num_nodes*num_nodes),
                                      bidirection=bidirection ) )
        elif ind < len(props.sl_model_params.split(",")):
            model.add( get_lstm_layer(units, 
                                      return_sequences=True,
                                      bidirection=bidirection) ) 
        else:
            model.add( get_lstm_layer(units, 
                                      bidirection=bidirection))
    model.add(Flatten())
    if demands:
        model.add(Dense(num_nodes**2, activation='linear') )
    else:
        model.add(Dense(env.get_num_tunnels(), activation='sigmoid') )
    return model




def lstm1(env, props, demands=True, bidirection=False):
    model = Sequential()
    num_nodes = env._num_nodes
    
    # iterate params to create rest of architecture
    for ind, params in enumerate(props.sl_model_params.split(",")):
        params = params.split(":")
        units = int(params[0]) 
        activation = params[1]
        if ind == 0:
            model.add( get_lstm_layer(units, 
                                      return_sequences=True, 
                                      input_shape=(props.hist_len, num_nodes*num_nodes),
                                      bidirection=bidirection ) )
        elif ind < len(props.sl_model_params.split(",")):
            model.add( get_lstm_layer(units, 
                                      return_sequences=True,
                                      bidirection=bidirection) ) 
        else:
            model.add( get_lstm_layer(units, 
                                      bidirection=bidirection))
    model.add(Flatten())
    if demands:
        model.add(Dense(1, activation='linear') )
    else:
        model.add(Dense(env.get_num_tunnels(), activation='sigmoid') )
    return model


def multi_arch(env, props, model_arch):
    num_nodes = env._num_nodes
    
    prev_input, model_prev = model_arch(props.hist_len, num_nodes, props)
    time_input, model_through_time = model_arch(props.look_len, num_nodes, props)
    
    output = concatenate([model_prev, model_through_time ])
    output = Dense(num_nodes*num_nodes, activation='relu')(output)
    output = Dense(num_nodes*num_nodes)(output)
    
    model = Model(inputs=[prev_input, time_input], outputs=[output])

    return model


def get_model(env, props, force_model = None):
    if props.sl_model_type == FCN:
        model = fcn(env, props)
    
    elif props.sl_model_type == CNN_1D_COMM:
        model = cnn1d_comm(env, props)

    elif props.sl_model_type == CNN_1D_DEMAND:
        model = cnn1d_demand(env, props)

    elif props.sl_model_type == CNN_2D_DEMAND:
        model = cnn2d_demand(env, props)

    elif props.sl_model_type == LSTM_DM:
        model = lstm(env, props, demands=True, bidirection=False)
    
    elif props.sl_model_type == LSTM_IMITATION:
        model = lstm(env,props, demands=False, bidirection=False)
    
    elif props.sl_model_type == BIDER_LSTM_DM:
        model = lstm(env, props, demands=True, bidirection=True)
    
    elif props.sl_model_type == BIDER_LSTM_IMITATION:
        model = lstm(env,props, demands=False, bidirection=True) 
    
    elif props.sl_model_type == FCN1:
        model = fcn1(env, props)    

    return model

