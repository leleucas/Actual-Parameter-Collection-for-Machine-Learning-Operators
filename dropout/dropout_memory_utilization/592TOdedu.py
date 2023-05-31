import json
import os
import ast
import numpy as np

samples = {}
with open("/home/pity/dropout/DL-Op-Analysis-main/Loss/mse/dropout_top592.json", "r") as f:
    # with open("/home/pity/dropout/DL-Op-Analysis-main/Loss/mse/mse_loss.json", "r") as f:
        samples = json.load(f)
samples_num = len(samples)

input_size = {'x1':[],'p':[],"training":[],"inplace":[]}
p_count = {}
for i in range(samples_num):
    if not(samples[i]['p'] == 0.0 or samples[i]['p'] == [0.0] or samples[i]['p'] == [[0.0]]):
        input_size['x1'].append(samples[i]['input_size'][0])
        p_tmp = samples[i]['p']
        while type(p_tmp) == type([]):
            p_tmp = p_tmp[0]
        input_size['p'].append(p_tmp)
        if p_count.get(str(p_tmp)) == None:
            p_count[str(p_tmp)] = 1
        else:
            p_count[str(p_tmp)] = p_count[str(p_tmp)]+1
        input_size["training"].append(samples[i]["training"][0])
        input_size["inplace"].append(samples[i]["inplace"][0])
with open("/home/pity/dropout/DL-Op-Analysis-main/Loss/mse/dropout_dedu.json", 'w') as input_json:
      json.dump(input_size, input_json)