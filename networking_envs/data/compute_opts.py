import os
import glob
import subprocess

opts_dir_prefix = 'opts_'
topology_name = 'GEANT'
paths_from = 'sp'

test_opts_dir = opts_dir_prefix + 'test'
train_opts_dir = opts_dir_prefix + 'train'

assert not os.path.exists(test_opts_dir)
assert not os.path.exists(train_opts_dir)

os.mkdir(test_opts_dir)
os.mkdir(train_opts_dir)

subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', topology_name, '--hist_len', '0', '--sl_type', 'stats_comm', '--compute_opts', '--paths_from', paths_from], check=True)
subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', topology_name, '--hist_len', '0', '--sl_type', 'eval', '--compute_opts', '--opts_dir', test_opts_dir], check=True)

subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', topology_name, '--hist_len', '0', '--sl_type', 'stats_comm', '--compute_opts', '--compute_opts_dir', 'train'], check=True)
subprocess.run(['python', '../../ml/sl_algos/evaluate.py', '--ecmp_topo', topology_name, '--hist_len', '0', '--sl_type', 'eval', '--compute_opts', '--compute_opts_dir', 'train', '--opts_dir', train_opts_dir], check=True)

for d in ['test', 'train']:
    opts_info = []
    for file in sorted(glob.glob(d + "/*.hist")):
        with open(file) as f:
            opts_info.append((file[:-4]+'opt', len(f.readlines())))
    
    input_file_idx = 0
    for i in range(len(opts_info)):
        opt_res_for_actual_demands = []
        for j in range(opts_info[i][1]):
            input_file_name = str(input_file_idx) + '.opt'
            with(open(opts_dir_prefix + d + '/' + input_file_name)) as f:
                lines = f.read().splitlines()
                for line in lines:
                    if line.startswith(' Optimal result for actual demand: '):
                        opt_res = float(line[line.find(': ')+1:])
                        opt_res_for_actual_demands.append(opt_res)
            
            input_file_idx += 1
            
        with open (opts_info[i][0], 'w') as f:
            for o in opt_res_for_actual_demands:
                f.write(str(o) + '\n')
