from gym.core import Env
from networking_env.environments.fake_spec import FakeSpec
import numpy as np

class BaseEnv(Env):
    ENVID = 0
    
    def __init__(self, props):
        self.props = props
        
        self._init()
        self._set_optimizer()
        self._set_simulator() 
        self._set_observation_space()
        self._set_action_space()
        self._spec = FakeSpec(props, self)
        self._envid = BaseEnv.ENVID
        BaseEnv.ENVID += 1
        
    def _set_observation_space(self):
        raise NotImplementedError("Please implement method")
    
    def _set_action_space(self):
        raise NotImplementedError("Please implement method")
    
    def log_diagnostics(self, paths, *args, **kwargs):
        pass
    
    @property
    def observation_space(self):
        return self._observation_space                                                                         

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation(self):
        return self._observation
    
    def get_num_edges(self):
        pass
    
    def get_num_nodes(self):
        pass
    
    def _init(self):
        pass
    
    def test(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
    
    def _set_simulator(self):
        pass
    
    def _set_optimizer(self):
        pass
