from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from click.core import batch
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 64
num_train = 4*batch_size*10000
num_test = 4*batch_size*1000
num_nodes = 24

train_cur_id = 0
test_cur_id = 0



def sample_tms(amnt):
    tms = []
    for _ in range(amnt):
        tms.append(np.random.rand(num_nodes,num_nodes))
    return tms

train = sample_tms(num_train)
test = sample_tms(num_test)


def get_batches(batch_size,values):
    batches = []
    batch_id = 0
    while batch_id*batch_size < len(values):
        end = min(len(values), (batch_id+1)*batch_size)
        tms = [values[v].flatten() for v in range(batch_id*batch_size, end)]
        batches.append( np.stack(tms) )
        batch_id+=1
    return batches

train_batches = get_batches(batch_size, train)
test_batches = get_batches(4, test)

train_batch_id = 0
test_batch_id = 0

def get_train_batch(batch_size):
    global train_batch_id
    batch = train_batches[train_batch_id]
    if train_batch_id + 1 < len(train_batches):
        train_batch_id += 1
    else:
        train_batch_id = 0
    return batch

def get_test_batch(batch_size):
    global test_batch_id
    batch = test_batches[test_batch_id]
    if test_batch_id + 1 < len(test_batches):
        test_batch_id += 1
    else:
        test_batch_id = 0
    return batch

# Training Parameters
learning_rate = 0.01
num_steps = 30000


display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_hidden_3 = 64
num_input = num_nodes**2 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    return layer_3


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    return layer_3

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
#         batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = get_train_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((num_nodes * n, num_nodes * n))
    canvas_recon = np.empty((num_nodes * n, num_nodes * n))
    t_loss = []
    for i in range(n):
        # MNIST test set
#         batch_x, _ = mnist.test.next_batch(n)
        batch_x = get_test_batch(batch_size)
        # Encode and decode the digit image
        lt= sess.run(decoder_op, feed_dict={X: batch_x})
        t_loss.append( lt )
    import pdb;pdb.set_trace()
#         # Display original images
#         for j in range(n):
#             # Draw the original digits
#             canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                 batch_x[j].reshape([28, 28])
#         # Display reconstructed images
#         for j in range(n):
#             # Draw the reconstructed digits
#             canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
#                 g[j].reshape([28, 28])

#     print("Original Images")
#     plt.figure(figsize=(n, n))
#     plt.imshow(canvas_orig, origin="upper", cmap="gray")
#     plt.show()
# 
#     print("Reconstructed Images")
#     plt.figure(figsize=(n, n))
#     plt.imshow(canvas_recon, origin="upper", cmap="gray")
#     plt.show()