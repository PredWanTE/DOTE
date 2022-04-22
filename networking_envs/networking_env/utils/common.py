import numpy as np
import random


def set_global_seeds(i, use_tf = False):
    if use_tf:
        try:
            import tensorflow as tf
            tf.set_random_seed(i)
        except:
            pass
    
    np.random.seed(i)
    random.seed(i)
