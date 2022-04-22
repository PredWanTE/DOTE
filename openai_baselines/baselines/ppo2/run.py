import sys
import os

cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

import tensorflow as tf

from networking_env.environments.ecmp.env_args_parse import parse_args
from networking_env.environments.ecmp import history_env
from networking_env.environments.ecmp.optimizers.path_optimizer import PathOptimizer
from networking_env.environments.consts import RLMode
from networking_env.utils.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, ortho_init
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers
from baselines.ppo2 import ppo2
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy

import numpy as np
import tensorflow as tf
import sys
import time
from copy import deepcopy
from tqdm import tqdm



def mlp_3(num_layers=5, num_hidden=128, activation=tf.tanh, layer_norm=False, time_data=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        with tf.variable_scope('y'):
            batch_size = X.get_shape()[0].value

            if time_data:
                X_without_time = X[:, 2:]

            if time_data:
                h = tf.layers.flatten(X_without_time)
            else:
                h = tf.layers.flatten(X)

        # h = tf.layers.flatten(h)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            if i < num_layers - 1:
                h = activation(h)
                h = tf.layers.dropout(h, rate=0.5)
            else:
                # in our last layer, connect the timestamps from the input
                if time_data is True:
                    interim = tf.reshape(X[:, :, :2], (batch_size, -1))
                    h = tf.concat([h, interim], axis=1)
                h_ = fc(h, 'mlp_fc_2_{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
                h_ = tf.sigmoid(h_)

        return h

    return network_fn


def train():
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()


    def make_env():
        import sys
        props = parse_args(sys.argv[1:])
        env = history_env.ECMPHistoryEnv(props)
        env.seed(props.seed)
        return env

    # make only one environment
    env = DummyVecEnv([make_env])

    props = parse_args(sys.argv[1:])
    set_global_seeds(props.seed)

    nsteps = 100
    nminibatches = 1
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5

    policy = mlp_3()
    if props.rl_mode == RLMode.TRAIN or props.rl_mode == RLMode.BOTH:
        print('***** TRAIN *****')
        print(props.num_epochs * props.train_batch)
        # import pdb;pdb.set_trace()
        model = ppo2.learn(network=policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
                   lam=0.95, gamma=0, noptepochs=10, log_interval=1,
                   ent_coef=ent_coef,
                   lr=3e-4,
                   cliprange=0.2,
                   total_timesteps=props.num_epochs * props.train_batch,
                   test_interval=100, save_interval=100)


        # save model
        model.save('model_rl.pkl')

    if props.rl_mode == RLMode.TEST or props.rl_mode == RLMode.BOTH:

        if RLMode.TEST:
            policy = build_policy(env, policy)
            model = Model(policy=policy,
                          ob_space=env.observation_space,
                          ac_space=env.action_space,
                          nbatch_act=env.num_envs,
                          nbatch_train=((env.num_envs * nsteps) / nminibatches),
                          nsteps=nsteps,
                          ent_coef=ent_coef,
                          vf_coef=vf_coef,
                          max_grad_norm=max_grad_norm)

            model.load('model_rl.pkl')

        # move to test env
        env.envs[0].test(True)
        # env = DummyVecEnv([make_env_test])
        # env = VecNormalize(env)

        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        print('***** TEST *****')

        res = []
        times = []
        s_time = time.time()
        test_timesteps = env.envs[0]._simulator._test_hist.num_tms()
        for i in tqdm(range(test_timesteps)):
            start_time = time.time()
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
            obs, rew, done, _ = env.step(actions)
            times.append(time.time() - start_time)
            res.append(rew[0])
            # print(i, rew[0])

            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        print('Total time: %f' % (time.time() - s_time))
        print('AVG reward %f' % np.mean(res))
        print('Total time steps %d' % test_timesteps)

        with open('rl_stats.txt', 'w') as f:
            import statistics
            dists = [-1.0*float(v) for v in res]
            dists.sort()
            f.write('Average: ' + str(statistics.mean(dists)) + '\n')
            f.write('Median: ' + str(dists[int(len(dists)*0.5)]) + '\n')
            f.write('75TH: ' + str(dists[int(len(dists)*0.75)]) + '\n')
            f.write('90TH: ' + str(dists[int(len(dists)*0.90)]) + '\n')
            f.write('95TH: ' + str(dists[int(len(dists)*0.95)]) + '\n')
            f.write('99TH: ' + str(dists[int(len(dists)*0.99)]) + '\n')

if __name__ == "__main__":
    train()
