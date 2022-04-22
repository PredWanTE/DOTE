from networking_env.utils.shared_consts import SizeConsts
import joblib
from subprocess import PIPE, Popen
import numpy as np
import shutil

def jar_cplex_wrapper(cplex_path, args, all_output=True):
    process = Popen(['java'] + cplex_path + ['-jar']+args, stdout=PIPE, stderr=PIPE)
    
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode('utf-8')
    if stderr.strip() != "":
        raise Exception(stderr)
    
    if all_output:
        return stdout.strip().split('\n')
    else:
        return float(stdout.strip().split('\n')[-1])


def jar_gurobi_wrapper(props, cplex_path, args, all_output=True):

    # change JAR file to get filename instead of actual params.
    # this is done because of PATH size limit
    with open('param.tmp', 'w') as file:
        for k in range(len(args)):
            param = args[k]
            if k == 0 or k == (len(args) - 1) or k == (len(args) - 2) or (props.compute_opts and k == (len(args) - 4)):
                continue
            if ((k == (len(args) - 4) or k == (len(args) - 3)) and param == ""): param = "0 1 0.0"
            file.write(str(param) + '\n')

    #create a copy of param.tmp in "opts"
    #shutil.copy("param.tmp", props.opts_dir + "/param" + str(args[-2]) + ".tmp")
    #return [': 1']
    process = Popen(['java'] + ['-jar'] + [args[0]] + ['param.tmp'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode('utf-8')

    n_tries = 0
    while stderr.strip() != "" and n_tries < 100:
        print(stderr.strip())
        n_tries += 1
        
        if 'optimze' in stderr.strip():
            return [': 1']
        import time;
        time.sleep(60)

        process = Popen(['java'] + ['-jar'] + [args[0]] + ['param.tmp'], stdout=PIPE, stderr=PIPE)

        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode('utf-8')
        
        if stderr.strip() != "" and n_tries == 100:
            joblib.dump(stderr, 'stderr.pkl')
            raise Exception(stderr)

    with open(str(args[-1]) + str(args[-2]) + '.opt', 'w') as f:
        f.write(' '.join(stdout.strip().split('\n')))

    if all_output:
        return stdout.strip().split('\n')
    else:
        return float(stdout.strip().split('\n')[-1])


def tm_to_string(tm):
    res = []
    import numpy as np
    num_nodes = int(np.sqrt(tm.shape[0]))
    tm = np.reshape(tm, (num_nodes, num_nodes))
    for i in range(tm.shape[0]):
        for j in range(tm.shape[0]):
            if i == j: continue
            if tm[i, j] > 0:
                res.append("%d %d %f" % (i, j, tm[i, j]))
    return ",".join(res)


def get_opt_cplex(props, tm, next_tm=None, opt_function="MAXUTIL", combine=None, use_cplex=False, idx=0, path=None):
    tunnels_file = '/'.join(props.g_name.split("/")[:-1]) + "/tunnels.txt"
    args_java = [props.jar_cplex_loc] if use_cplex else [props.jar_gurobi_loc]

    args_java += [opt_function, props.g_name, tunnels_file, tm_to_string(tm)]
    if next_tm is not None:
        args_java += [tm_to_string(next_tm)]
    if use_cplex:
        res = jar_cplex_wrapper(props.cplex_path, args_java)
    else:
        args_java += [idx]
        args_java += [path]
        res = jar_gurobi_wrapper(props, props.gurobi_path, args_java)
    return res if combine is None else combine.join(res)


def abeline_scale(tm):
    # tm is in unit of 100b/5minutes
    # we want to change that into Mb/s
    
    # first convert to b/s ==> 100b/5min = 100B/300s = 1B/3s
    # so our unit is 8/3 bps, we want to convert it to 1bps first
    tm = tm*(8/3)
    return tm/SizeConsts.ONE_Mb  # convert to MBps


def tms_list_to_file(fname, tms):
    with open(fname, 'w') as f:
        for tm in tms:
            tm_line = " ".join([str(_) for _ in tm.flatten()])
            f.write(tm_line+"\n")


def get_hists_from_folder(folder, limit = None):
    import glob
    no_limit = (limit is None) or (limit < 1)
    hists = sorted(glob.glob(folder + "/*.hist"))
    return hists if no_limit else hists[:limit]


def get_data_dir(props, is_test):
    postfix = "test" if is_test else "train"
    return props.hist_location+"/%s/%s/" % (props.ecmp_topo, postfix)


def get_data_dir_latent(props, is_test):
    postfix = "test" if is_test else "train"
    return props.hist_location + "/%s/%s/%s/" % (props.ecmp_topo, props.latent_dir, postfix)


def get_train_test_files(props):
    train_folder = get_data_dir(props, is_test=False)
    test_folder = get_data_dir(props, is_test=True)

    # these are the histories names 
    train_hist_files = get_hists_from_folder(train_folder)
    test_hist_files = get_hists_from_folder(test_folder)
    
    return sorted(train_hist_files), sorted(test_hist_files)


def get_train_test_files_latent(props):
    if props.rl_mode == "":
        return [], []
        
    train_folder = get_data_dir_latent(props, is_test=False)
    test_folder = get_data_dir_latent(props, is_test=True)

    # these are the histories names
    train_hist_files = get_hists_from_folder(train_folder)
    test_hist_files = get_hists_from_folder(test_folder)

    return sorted(train_hist_files), sorted(test_hist_files)