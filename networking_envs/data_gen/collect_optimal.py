from networking_env.environments.ecmp.env_args_parse import parse_args

import os
from tqdm import tqdm

from data_gen import utils as DGU
from ml.sl_algos import utils as SLU

def output_to_routing(res):
    if isinstance(res,list): res = '\n'.join(res)
    tunnels = [l for l in res.split("\n") if "Tunnel(" in l]
    tunnel_to_frac = [""]*len(tunnels)
    for tid, t in enumerate(tunnels):
        if not t.strip(): continue
#         tid = int( t.split("(")[1].split(")")[0] )
        frac = t.split(":")[-1].strip()
        tunnel_to_frac[tid] = frac
    opt_res = res.split("\n")[0].split(":")[1].strip()
    
    return ",".join(tunnel_to_frac), opt_res

def main(args):
    # do this on a file by file basis
    hist_name = args[0]
    train_test = args[1]
    props = parse_args(args[2:])
    
    base_folder = "%s/%s/"%(props.hist_location, props.ecmp_topo)
    
    fname = "%s/%s/%s"%( base_folder, train_test, hist_name)
    
    tms = SLU.get_data([fname], None)
    
    ############################
    # do regualr
    tunnel_frac = []
    opt_res = []
    for i, tm in enumerate(tqdm(tms)):
        res_str = DGU.get_opt_cplex(props, tm)
        tunnels, opt = output_to_routing(res_str)
        tunnel_frac.append(tunnels)
        opt_res.append(opt)
             
    with open(fname +".opt", 'w') as f:
        f.write('\n'.join(opt_res))
    with open(fname +".tunnels", 'w') as f:
        f.write('\n'.join(tunnel_frac))
    
    ############################ 


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
