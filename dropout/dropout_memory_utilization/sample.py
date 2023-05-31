import csv
import json
import os
import ast
import numpy as np
# import torch
# from sample_config import *
# from judge import *

#
# def get_sample_config():
#     with open("/home/pity/dropout/DL-Op-Analysis-main/Loss/mse/dropout_top592.json", "r") as f:
#         arg_data = json.load(f)
#     return SampleConfig(
#         args_cases=arg_data,
#         requires_grad=[False] * len(arg_data[0]),
#         backward=[False],
#         performance_iters=1,
#         save_timeline=False,
#         url="",  # noqa
#         tags=[],
#     )

def gen_np_args(x1_, x2_, reduction_):
    x1 = np.random.random(x1_).astype(np.float32)
    x2 = np.random.random(x2_).astype(np.float32)

    return [x1, x2, reduction_]


#
# def args_adaptor(np_args):
#     # x1 = torch.from_numpy(np_args[0]).cuda()
#     x1 = torch.from_numpy(np_args[0])
#     # x2 = torch.from_numpy(np_args[1]).cuda()
#     x2 = torch.from_numpy(np_args[1])
#     reduction_ = 'mean'
#     if np_args[2] != "":
#         if not argIsNone(np_args[2]):
#             if isinstance(np_args[2], list):
#                 reduction_ = np_args[2][0]
#             else:
#                 reduction_ = np_args[2]
#     return [x1, x2, reduction_]



def get_input_data():
    with open("/home/pity/dropout/DL-Op-Analysis-main/Loss/mse/mse_loss_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_data():
    with open("dropout_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_info():
    with open("dropout_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_infos = []
    for i in range(arg_data_length):
        in_info = str(arg_data["x1"][i][0])
        for dim in range(len(arg_data["x1"][i])-1):
            in_info += "x" + str(arg_data["x1"][i][dim+1])
        in_infos.append(in_info)
    return in_infos

def get_vritual_input_data():
    with open("./mse_loss_vritual.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim[0] # List
        in_sizes.append(in_size)
    return in_sizes

def get_vritual_input_info():
    with open("./mse_loss_vritual.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_infos = []
    for i in range(arg_data_length):
        in_infos.append("2^"+str(i+1))
    return in_infos