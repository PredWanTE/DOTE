import os
from networking_env.utils.shared_consts import FolderPathCosts
from networking_env.environments.ecmp.optimizers import consts as OC
from networking_env.utils import logger
from networking_env.utils.properties import Properties
from networking_env.environments.consts import ActionConsts, ActionPostProcess

def add_default_args(parser):
    
    # Environement arguments
    parser.add_argument("--flatten_hist", action="store_true")
    parser.add_argument('--env_layers', type=str, default="")
    parser.add_argument("--opts_dir", type=str, default="opts")
    
    # Simulator arguments
    parser.add_argument("--history_action_type", type=str, default=ActionConsts.ACTION_PATHS_SPLITING_RATIOS) # ACTION_W_EPSILON
    parser.add_argument("--update_every", type=int, default=5, help="update every K minutes")
    parser.add_argument('--num_test_histories', type=int, default=1)
    parser.add_argument('--num_train_histories', type=int, default=1)
    parser.add_argument('--no_dump', action="store_true")
    parser.add_argument('--time', action="store_true")
    parser.add_argument("--compute_opts", action="store_true")
    parser.add_argument("--compute_opts_dir", type=str, default="test")

    # Topology arguments
    parser.add_argument("--ecmp_topo", type=str, default="TEST")
    parser.add_argument("--max_weight", type=float, default=50.0)
    parser.add_argument("--ecmp_version",type=int, default=1)
    
    # Topology optimizer arguments
    parser.add_argument("--optimization_steps", type=int, default=500)
    parser.add_argument("--optimizer_class", type=str, default=OC.PATH_OPTIMIZER)
    parser.add_argument("--paths_from", type=str, default=OC.PATHS_FROM_FILE)
    parser.add_argument("--nx_graph", action="store_true")
    parser.add_argument("--from_routing", action="store_true")
    parser.add_argument("--num_tunnels", type=int, default=8)
    parser.add_argument("--paths_file", type=str, default="tunnels.txt")
    parser.add_argument("--opt_function", type=str, default="MAXUTIL")

    # Other arguments
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--generate_baselines", action="store_true")
    parser.add_argument("--generate_supervised", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--hist_type", type=str, default="sigcomm18")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument('--dump_policy_every', type=int, default=1000)
    
    # TM arguments
    parser.add_argument("--tm_type", type=str, default="abilene")
    parser.add_argument("--big_flow_size_mb", type=float, default=4000)
    parser.add_argument("--elphnt_cap_perc", type=float, default=0.4)
    parser.add_argument("--small_flow_size_mb", type=float, default=100)
    parser.add_argument("--mice_cap_perc", type=float, default=0.6)
    parser.add_argument("--elephant_load", type=float, default=0.4)
    parser.add_argument("--sparse_level", type=float, default=0.3)

    ############ RL arguments #############
    # General RL arguments
    parser.add_argument("--action_postprocess", type=str, default="scale_env")
    parser.add_argument("--history_reward_type", type=str, default="avg")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--layers", type=str, default="32,32")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--decay_strat', type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--iter_per_test", type=int, default=5)
    parser.add_argument('--action_process', type=str, default=ActionPostProcess.SCALE_TO_ENV)
    parser.add_argument('--ae', action="store_true")
    parser.add_argument('--latent_dir', type=str, default=".")
    parser.add_argument('--latent_dim', type=int, default=15)
    parser.add_argument('--output_dim', type=int, default=100)
    parser.add_argument('--rl_mode', type=str, default="")
    
    ############ DOTE arguments #############
    parser.add_argument('--so_mode', type=str, default="train")
    parser.add_argument("--so_epochs", type=int, default=1)
    parser.add_argument("--so_batch_size", type=int, default=1)
    parser.add_argument("--so_max_conc_batch_size", type=int, default=50)

    # TRPO arguments
    parser.add_argument("--max_path_length", type=int, default=50)
    parser.add_argument("--num_paths", type=int, default=20)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument('--run_direct', action="store_true")
    parser.add_argument('--resume_from', type=str, default="")
    parser.add_argument('--snapshot_mode', type=str, default="all")
    parser.add_argument('--additional_iterations', type=int, default=1)
    parser.add_argument('--policy_type', type=str, default="pg_cont_mlp")
    parser.add_argument('--tensorflow', type=bool, default="true")
    parser.add_argument('--adaptive_std', type=bool, default="true")
    
    # ACKTR arguments
    parser.add_argument('--desired_kl', type=float, default=0.002)
    parser.add_argument('--sgd_step', type=float, default=0.03)
    parser.add_argument("--vf_lr", type=float, default=0.001)
    
    ############ SL arguments #############
    parser.add_argument('--sl_type', type=str, default='')
    parser.add_argument('--use_imitation', action="store_true")
    
    # General SL arguments
    parser.add_argument("--sl_batch_size", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=-1)
    parser.add_argument("--num_edges", type=int, default=-1)
    parser.add_argument("--sl_model_type", type=str, default="linear_regression")
    parser.add_argument("--sl_optimizer" ,type=str, default="adam")
    parser.add_argument("--early_stopping", type=int, default=5)
    
    # SL layers
    parser.add_argument("--sl_model_params", type=str, default="150:relu,100:relu,50:relu")
    
    
    ############### TM sampling arguments ###################
    parser.add_argument("--tm_sample_rate_min", type=int, default=5)
    # this assumes that we sample 12 TMs per hour (1 per 5min)
    parser.add_argument('--hist_len', type=int, default=12, help="lookback on past TMs")
    
    # now we want to define lookahead/behind
    parser.add_argument('--look_len', type=int, default=3,
                        help="how many samples to take for each TM in the lookbehind/ahead")
    parser.add_argument("--tm_hour_lookbehind", type=int, default=6)
    parser.add_argument("--tm_hour_lookahead", type=int, default=6)
    parser.add_argument('--scale_demands', type=float, default=1.0, help="Scale the TMs by this factor")
    parser.add_argument("--sl_src", type=int, default=-1)
    parser.add_argument("--sl_dst", type=int, default=-1)
    
    
    return parser


def process_args(args):
    
    if args.ecmp_topo is None:
        raise Exception("If using ECMP, must add the topology file location")

    # remote - running on remote machine
    # not remote - running on local machine
    if args.remote:
        base_path = FolderPathCosts.BASE_PATH_REMOTE
        cplex_path = r"PATH_TO_CPLEX_REMOTE"
        gurobi_path = r"PATH_TO_GUROBI_REMOTE"
        oblivious_loc = r'PATH_TO_OBLIVIOUS_REMOTE'
    else:
        #base_path = FolderPathCosts.BASE_PATH_LOCAL 
        base_path = os.getcwd()[:os.getcwd().find("networking_envs")] + "networking_envs"
        cplex_path = r"PATH_TO_CPLEX_LOCAL"
        gurobi_path = r"PATH_TO_GUROBI_LOCAL"
        oblivious_loc = r'PATH_TO_OBLIVIOUS_LOCAL'
    
    jar_cplex_loc = base_path + "/lib/runner_cplex.jar"
    jar_gurobi_loc = base_path + "/lib/runner_gurobi.jar"

    graph_path = "%s/data/" % base_path
    hist_path = "%s/data/" % base_path
    
    args.base_path = base_path
    args.graph_base_path = graph_path
    args.jar_cplex_loc = jar_cplex_loc
    args.jar_gurobi_loc = jar_gurobi_loc
    args.oblivious_loc = oblivious_loc
    cplex_path = ("-Djava.library.path=%s -ea" % cplex_path).split(" ")
    gurobi_path = ("-Djava.library.path=%s -ea" % gurobi_path).split(" ")
    # set the graph path
    
    args.hist_location = hist_path
    
    args.cplex_path = cplex_path
    args.gurobi_path = gurobi_path

    graph_name = args.ecmp_topo + "/" + args.ecmp_topo + '_int.pickle.nnet'
    args.graph_path = graph_path + graph_name
    
    # no need to define other variables as those depend
    # on the simulations
    if args.generate_baselines or args.generate_data:
        return

    args.layers = tuple(map(int,args.layers.split(",")))
    
#     if args.env_layers:
#         layers = tuple(map(float, args.env_layers.split(",")))
#         args.layers = tuple( int(np.ceil(sz*env.get_g().get_num_edges())) for sz in layers )
    is_sl = args.sl_type != ""
    args.dir_name = get_result_dir(args, is_sl)
    if not args.debug:
        os.makedirs(args.dir_name, exist_ok=True)
        # log in all formats
        logger.reset()
        logger.configure(args.dir_name, logger.LOG_OUTPUT_FORMATS)

    args.exp_name = "base_exp"
    
    # Augment other RL params
    args.step_size = args.lr # for rllab TRPO
    args.n_iter = args.num_epochs # for rllab TRPO
    args.g_name = "%s/%s/%s_int.txt" % (args.graph_base_path, args.ecmp_topo, args.ecmp_topo)
    return args


def get_result_dir(props, is_sl):
    path_ = [props.hist_location,
                props.ecmp_topo, 
                #"scale_%.1f"%props.scale_demands,
                "models_ignore"]
    path_ += ['sl'] if is_sl else ['rl']
    path_ += ["hist_len=%d"%props.hist_len]
    return '/'.join(path_)


def get_result_dir_old(args):
    import datetime
    import dateutil.tz
    
    tm_params="p:%s"%str(args.sparse_level)

    dir_params = [args.base_path,
                  "results",
                  "sigcomm18",
                args.ecmp_topo,
#                     args.history_action_type,
#                     "train="+str(args.num_train_histories),
                args.tm_type, 
                "params-"+tm_params,
#                     "sparsity=%s"%args.p,
#                     args.policy_type.replace("_","-"), 
                "layers-"+str(args.layers), 
#                     str(args.max_path_length), 
#                     "num_path="+str(args.num_paths),
#                     str(args.num_paths),
                "action_post-"+args.action_postprocess,
                "seed-%d"%args.seed,
                ]
    
    if "bimodal" in args.tm_type:
        dir_params += ["elephant_load-%s"%str(args.elephant_load)]
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    dir_params.append(timestamp)
    return '/'.join(dir_params)
   
    
def parse_args(args):
    
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    args_ = parser.parse_args(args)
    
    # add additional metadata
    args_ = process_args(args_)
#     logger.log(args)
    return Properties(args_)
