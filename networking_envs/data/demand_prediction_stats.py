import os
import statistics

opts_dir = 'opts'
output_file_name = 'dp_stats.txt'

count = 0

dists = []
for file in os.listdir(opts_dir):
    if file.endswith('.tmp'): continue #param.tmp
    assert file.endswith('.opt')
    with open(opts_dir + '/' + file) as f:
        lines = f.read().splitlines()
        dist = -1.0
        for line in lines:
            if not line.startswith(' How far from optimal: '): continue
            dist = float(line[line.find(': ')+1:])
            break
        if dist == -1: print(file)
        assert dist != -1.0
        dists.append(dist)
        count += 1

dists.sort()
with open(output_file_name, 'w') as f:
    f.write('Average: ' + str(statistics.mean(dists)) + '\n')
    f.write('Median: ' + str(dists[int(len(dists)*0.5)]) + '\n')
    f.write('75TH: ' + str(dists[int(len(dists)*0.75)]) + '\n')
    f.write('90TH: ' + str(dists[int(len(dists)*0.90)]) + '\n')
    f.write('95TH: ' + str(dists[int(len(dists)*0.95)]) + '\n')
    f.write('99TH: ' + str(dists[int(len(dists)*0.99)]) + '\n')

#print("NUMBER OF FILES: " + str(count))
