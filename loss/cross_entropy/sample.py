import csv
import json
import os
import ast
import numpy as np
import torch
from sample_config import *
from judge import *

def get_sample_config():
    with open("./cross_entropy.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["target"][i], 
            arg_data["weight"][i], arg_data["size_average"][i], arg_data["ignore_index"][i], 
            arg_data["reduce"][i], arg_data["reduction"][i], arg_data["label_smoothing"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 8,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        url="",  # noqa
        tags=[],
    )

def gen_np_args(input_, target_, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_):
    input = np.random.random(input_).astype(np.float32)
    assert input_[1] > 0, "reduce size can not < 0"
    target = np.random.randint(input_[1], size=target_)

    return [input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_]

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    weight_ = None
    size_average_ = None
    ignore_index_ = -100
    reduce_ = None
    reduction_ = 'mean'
    label_smoothing_ = 0.0
    if np_args[2] != "":
        if np_args[2] == 'null':
            weight_ = None
        elif isinstance(np_args[2], list):
            if len(np_args[2]) == 1:
                weight_ = np_args[2][0]
            elif len(np_args[2]) == 2:
                weight_ = torch.tensor(np_args[2][0], device=torch.device(np_args[2][1]))
    if np_args[3] != "":
        if not argIsNone(np_args[3]):
            size_average_ = np_args[3] #TODO maybe list
    if np_args[4] != "":
        if not argIsNone(np_args[4]):
            ignore_index_ = np_args[4][0][0] #TODO maybe list
    if np_args[5] != "":
        if not argIsNone(np_args[5]):
            reduce_ = np_args[5] #TODO maybe list
    if np_args[6] != "":
        if not argIsNone(np_args[6]):
            reduction_ = np_args[6] #TODO maybe list
    if np_args[7] != "":
        if not argIsNone(np_args[7]):
            label_smoothing_ = np_args[7] #TODO maybe list
 
    return [input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_]

def get_input_data():
    with open("./cross_entropy.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    in_sizes = []
    label_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input"][i]:
            in_size *= dim
        in_sizes.append(in_size)
        label_sizes.append(arg_data["input"][i][1])
    return in_sizes, label_sizes

def get_dedu_input_data():
    with open("./cross_entropy_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    in_sizes = []
    label_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input"][i]:
            in_size *= dim
        in_sizes.append(in_size)
        label_sizes.append(arg_data["input"][i][1])
    return in_sizes, label_sizes


def get_dedu_target_data():
    with open("./cross_entropy_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["target"])
    target_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["target"][i]:
            in_size *= dim
        target_sizes.append(in_size)
    return target_sizes

def get_dedu_input_info():
    with open("./cross_entropy_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    in_infos = []
    for i in range(arg_data_length):
        in_info = str(arg_data["input"][i][0])
        for dim in range(len(arg_data["input"][i])-1):
            in_info += "x" + str(arg_data["input"][i][dim+1])
        in_infos.append(in_info)
    return in_infos

def get_dedu_target_info():
    with open("./cross_entropy_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["target"])
    in_infos = []
    for i in range(arg_data_length):
        in_info = str(arg_data["target"][i][0])
        for dim in range(len(arg_data["target"][i])-1):
            in_info += "x" + str(arg_data["target"][i][dim+1])
        in_infos.append(in_info)
    return in_infos