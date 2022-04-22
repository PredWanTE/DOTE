class FakeSpec:
    # a way to create a fake spec from properties file
    def __init__(self, props, env):
        for k,v in props.__dict__.items():
            self.add(k, v)
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def add(self, k, v):
        self.__dict__[k] = v
        
    @property
    def timestep_limit(self):
        return self.max_path_length - self.hist_len