import sys
import os

cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

from networking_env.environments.ecmp.env_args_parse import parse_args
from ml.sl_algos.utils import get_train_test
from ml.sl_algos import utils as SLU
 
from ml.sl_algos.stats import evaluate as StatsEval
from ml.ae_algos import evaluate as AEEval
from networking_env.utils.common import set_global_seeds

IMITATION_GEN = "imitation"
SL_NN_DEMAND = "nn_demand"
SL_NN_COMMODITY = "nn_comm"
SL_STATS = "stats_comm"
SL_STATS_EVAL = "eval"

AUTO_ENCODER = "ae"


class Bla():
    def __init__(self):
        self._num = 135

    def get_num_nodes(self):
        return self._num


def main(args):
    props = parse_args(args)
    set_global_seeds(props.seed)
    if props.sl_type == SL_STATS_EVAL:
        SLU.compute_cplex_res(props) 
    else:
        from networking_env.environments.ecmp import history_env
        from ml.sl_algos.nn import evaluate as NNEval
        env = history_env.ECMPHistoryEnv(props)
        env.seed(props.seed)

        if props.sl_type == IMITATION_GEN:
            train, test, extra_data = get_train_test(props)
        
        elif props.sl_type == SL_NN_DEMAND:
            train, test, extra_data = get_train_test(props)
            NNEval.main(env, props, train[0], test[0], extra_data)
            
        elif props.sl_type == AUTO_ENCODER:
            train, test, extra_data = get_train_test(props, lambda tms, props: (tms, tms))
            AEEval.main(env, props, train[0], test[0])

        elif "comm" in props.sl_type:
            from functools import partial
            print('\n[+] Processing training data...\n')
            train_all, test_all, extra_data = get_train_test(props, 
                                                                partial(SLU.basic_per_commodity,
                                                                num_nodes=env.get_num_nodes()))
            print('[+] Finished processing training data.\n')
            print('[+] Start training...\n')
            StatsEval.main(env, props, train_all, test_all, extra_data)

        else:
            raise NotImplementedError("We only support predictions based on NNs or standard statistical models")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])