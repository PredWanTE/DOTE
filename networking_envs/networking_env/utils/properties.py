class Properties(object):
    '''
    This class contains a generic properties file
    '''
    CURRENT = None

    def __init__(self, args):
        self._is_empty = True
        self._update(args)

    def __str__(self):
        header = "*"*5 + " Properties File " + "*"*5
        res = header + "\n"
        for key in self.__dict__:
            res += " "*5 + "%s\t:\t%s\n"%(str(key), str(self.__dict__[key]))
        res += "*"*len(header) + "\n"
        return res

    def __contains__(self, key):
        return key in self.__dict__

    def dump(self, fname): pass
#         for key in sorted(self.__dict__.keys()):
            

    def get(self, key):
        return self.__dict__[key]

    def __in__(self, key):
        return key in self.__dict__
    
    def _update(self, args):
        if isinstance(args,dict):
            for key in args:
                self.__dict__[key] = args[key]
                self._is_empty = False
        else:
            for key in args.__dict__:
                self.__dict__[key] = args.__dict__[key]
                self._is_empty = False

    def is_empty(self): return self._is_empty

    @property
    def steps_per_hist(self):
        return self.num_paths * self.max_path_length * self.n_parallel

    @property
    def train_batch(self):
        return self.num_train_histories * self.steps_per_hist
#     args.train_batch = args.num_train_histories * steps_per_hist

    @property
    def test_batch(self):
        return self.num_test_histories * self.steps_per_hist
