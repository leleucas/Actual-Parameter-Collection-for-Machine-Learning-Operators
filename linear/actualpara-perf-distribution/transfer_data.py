import numpy as np
import json

def format_data(file):
    with open(file, "r") as f:
        samples = json.load(f)
    samples_num = len(samples)
    in_list = []
    out_list = []
    n_list = []
    in_set = {}
    out_set = {}
    n_set = {}
    for i in range(samples_num):
        case_args = samples[i]
        input_size = case_args['Input'][0]
        weight_size = case_args['Weight'][0]
        if len(weight_size) == 1:
            in_features = weight_size[0]
            out_features = 1
        else:
            out_features = weight_size[0]
            in_features = weight_size[1]
        n = 1
        if len(input_size) > 1:
            for i in range(len(input_size)-1):
                n *= input_size[i]
        in_list.append(in_features)
        out_list.append(out_features)
        n_list.append(n)

        if in_features not in in_set.keys():
            in_set[in_features] = 1
        else:
            in_set[in_features] += 1
        if out_features not in out_set.keys():
            out_set[out_features] = 1
        else:
            out_set[out_features] += 1
        if n not in n_set.keys():
            n_set[n] = 1
        else:
            n_set[n] += 1
        
    json_data = {}
    json_data['N'] = n_list
    json_data['cin'] = in_list
    json_data['cout'] = out_list

    with open('linear_para.json', 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    format_data('linear.json')