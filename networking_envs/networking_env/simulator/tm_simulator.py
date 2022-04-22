from data_gen import utils as DU
from networking_env.simulator.history import Histories


class TMSimulator(object):
    
    def __init__(self, props, num_nodes, ttl_capacity):
        self._props = props
        self._num_nodes = num_nodes
        self._ttl_capacity = ttl_capacity
        self._init_histories(props)
        self.set_test(False)
    
    def _init_histories(self, props):
        train_hist_files, test_hist_files = DU.get_train_test_files(props)
        train_hist_files_latent, test_hist_files_latent = DU.get_train_test_files_latent(props)
        self._train_hist = Histories(train_hist_files, "train",
                                     self._num_nodes,
                                     props.time,
                                     train_hist_files_latent,
                                     max_steps=self._props.max_path_length + self._props.hist_len)
        self._test_hist = Histories(test_hist_files, "test",
                                    self._num_nodes,
                                    props.time,
                                    test_hist_files_latent,
                                    max_steps=self._props.max_path_length + self._props.hist_len)

        props.num_train_histories = self._train_hist.num_tms()
        props.num_test_histories = self._test_hist.num_tms()
    
    def get_time(self):
        return self._cur_time
    
    def set_test(self, val):
        self._cur_hist = self._test_hist if val == True else self._train_hist
        if val is True:
            self._cur_hist.reset()
        
    def reset(self):
        self._cur_time = 0
        self._cur_hist.reset()

        # now create first observation out of the first hist_len TMs
        return [self.next_latent()[0] for _ in range(self._props.hist_len)]
    
    def update_current_observation(self, obs):
        self._cur_observation.pop(0)
        self._cur_observation.append(obs)
        
    def is_time(self, t):
        return self._cur_time == t
    
    def is_terminal(self):
        return False

    def next_tm(self):
        tm, tm_time, opt = self._cur_hist.get_next()
        return tm, tm_time, opt

    def next_latent(self):
        tm, tm_time, opt = self._cur_hist.get_next_latent()
        self._cur_time += tm_time
        return tm, tm_time, opt