import numpy as np
from ml.sl_algos import utils as SLU
from tqdm import tqdm

from ml.sl_algos.imitation import evolution_strategies as ES
from networking_env.environments.ecmp.env_args_parse import parse_args
from networking_env.utils.common import set_global_seeds
from networking_env.environments.ecmp import history_env

def main(args):
    hist_name = args[0]
    train_test = args[1]
    props = parse_args(args[2:])
    
    base_folder = "%s/%s/"%(props.hist_location, props.ecmp_topo)
    
    fname = "%s/%s/%s"%( base_folder, train_test, hist_name)
    
    tms = SLU.get_data([fname])
    opts = [np.float64(l.strip()) for l in open(fname+".opt").readlines() if l.strip() ]
    
    set_global_seeds(props.seed)
    
    env = history_env.ECMPHistoryEnv(props)
    env.seed(props.seed)
    env.reset()
    
    optimizer = env._optimizer
    res = []
    for tm, optv in tqdm(zip(tms,opts)):
        reward_w, w, _ = ES.find_w(optimizer, tm, optv, env.action_space.shape[0])
        res.append("%s\t%s\t%s"%(",".join(str(v) for v in w), str(reward_w), str(reward_w/optv) ) )
    with open(fname+".imitation.%s"%props.history_action_type, 'w') as fim:
        fim.write('\n'.join(res))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])