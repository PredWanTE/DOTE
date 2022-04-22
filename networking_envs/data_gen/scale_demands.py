from data_gen import utils as datagen_utils
from networking_env.environments.ecmp.env_args_parse import parse_args

import os
from tqdm import tqdm
import numpy as np

from data_gen import utils as DU
from ml.sl_algos import utils as SLU

def random_scale(tm, mean):
    scale_val = np.random.normal(loc=mean, scale=1.5)
    if scale_val <=0: scale_val = np.random.rand()*mean
    return tm*mean

def main(args):
    props = parse_args(args)
    
    base_folder = "%s/%s/"%(props.hist_location, props.ecmp_topo)
    
    train_hist_files, test_hist_files = DU.get_train_test_files(props)
    
    scales = list(np.arange(2,10.5,0.5)) + list(range(10,21,2)) + list(range(30,101,10))
    scales = list(map(float,scales))
    
    def scale_files(hists, postfix, scale_func = lambda x,y: x*y, added_to_folder=""):
        for scale in tqdm(scales):
            folder = base_folder + "/scale%s_%.1f"%(added_to_folder,scale) + "/%s"%postfix 
            os.makedirs(folder, exist_ok=True)
            for hist in hists:
                tms = SLU.get_data([hist], None)
                hist_name = hist.split("/")[-1]
                with open(folder+"/"+hist_name, 'w') as f:
                    for tm in tms:
                        tmf = scale_func(tm.flatten(), scale)
                        f.write(" ".join([str(_) for _ in tmf]))
                        f.write("\n")
                        
    scale_files(train_hist_files, "train")
    scale_files(test_hist_files, "test")
     
    scale_files(train_hist_files, "train", random_scale, added_to_folder="_norm")
    scale_files(test_hist_files, "test", random_scale, added_to_folder="_norm")
     
     
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
