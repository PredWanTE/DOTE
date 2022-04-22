from gym import spaces
import networkx

from networking_env.environments.base import BaseEnv
from networking_env.environments.consts import RewardType, EdgeConsts, \
    ActionConsts, HistoryConsts
from networking_env.environments.ecmp import utils as ECMP_UTILS
from networking_env.environments.ecmp.utils import transform_action
from networking_env.environments.ecmp.optimizers.w_optimizer import WOptimizer
import numpy as np
from networking_env.simulator.tm_simulator import TMSimulator
from networking_env.environments.ecmp.optimizers import consts as OC
from networking_env.environments.ecmp.optimizers.path_optimizer import PathOptimizer
from networking_env.utils.shared_consts import SizeConsts


class ECMPHistoryEnv(BaseEnv):
    
    def init(self, props): 
        super(BaseEnv, self).__init__(props)
    
    def _set_observation_space(self):
        if self.props.ae:
            self._observation_space = spaces.Box(low=0.0,
                                                 high=np.inf,
                                                 shape=(self._history_len * self.props.latent_dim, ),
                                                 dtype=np.float32)
        else:
            if self.props.time is True:
                time = 2
            else:
                time = 0
            if self.props.flatten_hist:
                self._observation_space = spaces.Box(low=0.0,
                                                     high=np.inf,
                                                     shape=(self._history_len *
                                                            ((self._num_nodes *
                                                            self._num_nodes) + time),),
                                                     dtype=np.float32)
            else:
                self._observation_space = spaces.Box(low=0.0,
                                                     high=np.inf,
                                                     shape=(self._history_len, self._num_nodes * (self._num_nodes ) + time),
                                                     dtype=np.float32)

    def set_observation_space(self):
        self._set_observation_space()

    def _set_action_space(self):
        if self.props.history_action_type in ActionConsts.ACTIONS_W:
            self._action_space = spaces.Box(low=1, high=self.props.max_weight, shape=(self._num_edges, ),
                                            dtype=np.float32)
        elif self.props.history_action_type == ActionConsts.ACTION_SPLITTINT_RATIOS:
            self._action_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self._num_nodes * self._num_edges, ),
                                            dtype=np.float32)
        elif self.props.history_action_type == ActionConsts.ACTION_TM:
            self._action_space = spaces.Box(low=HistoryConsts.EPSILON, high=float('inf'),
                                            shape=(self._num_nodes * self._num_nodes, ),
                                            dtype=np.float32)
        elif self.props.history_action_type == ActionConsts.ACTION_PATHS_SPLITING_RATIOS:
            self._action_space = spaces.Box(low=HistoryConsts.EPSILON, high=1.0,
                                            shape=(self._optimizer._num_paths, ),
                                            dtype=np.float32)

    def set_action_space(self):
        self.set_action_space()

    def _init(self):
        self._g_name = self.props.g_name
        
        if not self.props.graph_path:
            g = ECMP_UTILS.get_base_graph()
        elif isinstance(self.props.graph_path, str):
            g = ECMP_UTILS.load_graph_from_file(self.props.graph_path)
        else:
            raise("We only support None or graph path") # networkx graphs,

        self._g = g
        
        # now running init for the graph generation
        self._num_nodes = networkx.number_of_nodes(g)
        self._history_len = self.props.hist_len
        
        self._adj = np.zeros((self._num_nodes, self._num_nodes))
        self._weighted_adj = np.zeros((self._num_nodes, self._num_nodes))
        self._capacities = []
        for s in range(self._num_nodes):
            for t in range(self._num_nodes):
                if s == t:
                    continue
                if t in self._g[s]:
                    cap = self._g[s][t][EdgeConsts.CAPACITY_STR]
                    cap = SizeConsts.BPS_TO_GBPS(cap)
                    self._adj[s, t] = 1.0
                    self._weighted_adj[s, t] = cap
                    self._capacities.append(cap)

        self._num_edges = len(self._capacities)
        self._capacities = np.array(self._capacities)
        self._num_tunnels = 0

    def init(self):
        self._init()
        
    def get_num_edges(self):
        return self._num_edges

    def get_num_nodes(self):
        return self._num_nodes
        
    def get_num_tunnels(self):
        return self._num_tunnels
    
    def test(self, val):
        self._simulator.set_test(val)
    
    def _set_simulator(self):
        self._simulator = TMSimulator(self.props, self._num_nodes, np.sum(self._capacities))

    def set_simulator(self):
        self._set_simulator()
    
    def _set_tunnels(self):
        import glob
        graph_folder = "/".join(self.props.graph_path.split("/")[:-1])

        tunnel_files = sorted(glob.glob(graph_folder + "/tunnels/*.tunnels"))
        for f in tunnel_files:
            self._num_tunnels += len([l for l in open(f).readlines() if l])
         
        self._tunnel_map = np.zeros((self._num_nodes * (self._num_nodes - 1), self._num_tunnels))
        cur_tid = 0
        for commid, f in enumerate(tunnel_files):
            comm_tunnels = len([l for l in open(f).readlines() if l])
            for commid_t in range(cur_tid, cur_tid + comm_tunnels):
                self._tunnel_map[commid, commid_t] = 1

    def set_tunnels(self):
        self.set_tunnels()
        
    def _set_optimizer(self):
        if self.props.optimizer_class == OC.WEIGHT_OPTIMIZER:
            optimizer_class = WOptimizer
        elif self.props.optimizer_class == OC.PATH_OPTIMIZER:
            optimizer_class = PathOptimizer
        
        self._optimizer = optimizer_class(self._adj, 
                                          self._capacities,
                                          self._weighted_adj,
                                          self.props)
        
        if self.props.history_action_type == ActionConsts.ACTION_SPLITTINT_RATIOS:
            self._optimizer.direct_q_value()
            
        if self.props.optimizer_class == OC.PATH_OPTIMIZER:
            self._set_tunnels()

    def set_optimizer(self):
        self._set_optimizer()
            
    def _transform_action(self, action):
        return transform_action(self, action)
        
    def _step(self, action):
        # when is the next time we reconfigure?
        next_time = self._simulator.get_time() + self.props.update_every
        rewards = []
        active_time = []
        observed_tms = []
        over_opt = []

        action_transformed = action #transform_action(self,action)
        if np.any(np.isnan(action_transformed)):

            raise ValueError("Action has nan values....")

        while not self._simulator.is_time(next_time):
            tm, _, _ = self._simulator.next_tm()
            latent, tm_time, opt_val = self._simulator.next_latent()
            rewards.append(self._optimizer.step(action_transformed, tm))
            active_time.append(tm_time)
            observed_tms.append(latent)
            rew_over_opt = 1.0 if opt_val == 0.0 else np.abs(rewards[-1]) / opt_val
            over_opt.append(rew_over_opt)

        current_reward = self._get_reward(over_opt, active_time)

        if np.isnan(current_reward) or current_reward <= 0:
            import pdb; pdb.set_trace()
        is_terminal = self._simulator.is_terminal()
        env_data = {'episode': {'r': current_reward, 'l': 1}}

        return self._get_observation(observed_tms), \
                    -1 * current_reward, \
                    is_terminal, \
                    env_data

    def step(self, action):
        return self._step(action)

    def _process_observation(self, observed_tms):
        return observed_tms[-1]

    def _get_observation(self, observed_tms):
        processed_current = self._process_observation(observed_tms)
        
        self._observations_hist.pop(0)
        self._observations_hist.append( processed_current )
        
        return self._observation
    
    @property
    def _observation(self):
        obs = np.vstack(self._observations_hist)
        obs = obs.flatten() if self.props.flatten_hist else obs
        return obs
    
    def _reset(self):
        self._observations_hist = self._simulator.reset()
        return self._observation

    def reset(self):
        return self._reset()
    
    def _get_reward(self, rewards, active_time):
        if len(rewards) == 1:
            return rewards[0]
        elif self.props.history_reward_type == RewardType.AVG:
            return np.average(rewards)
        elif self.props.history_reward_type == RewardType.MEAN:
            return np.average(rewards)
        elif self.props.history_reward_type == RewardType.MAX:
            return max(rewards)
        elif self.props.history_reward_type == RewardType.WEIGHTED_AVG:
            return np.average(rewards, weights=np.array(active_time) / sum(active_time))
