import csv
import json
import os
import ast
import numpy as np
import torch

def gen_np_args(x1_, x2_, reduction_):
    x1 = np.random.random(x1_).astype(np.float32)
    x2 = np.random.random(x2_).astype(np.float32)

    return [x1, x2, reduction_]

def get_dedu_input_data(average_len = 35):
    with open("interpolate_all_top5740_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length // average_len):
        in_size = 1
        for j in range(i * average_len, (i+1) * average_len):
            for dim in arg_data["x1"][j]:
                in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_info(average_len = 35):
    with open("interpolate_all_top5740_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_infos = []
    for i in range(arg_data_length // average_len):
        in_info = str(arg_data["x1"][i * average_len][0])
        for dim in range(len(arg_data["x1"][i * average_len])-1):
            in_info += "x" + str(arg_data["x1"][i * average_len][dim+1])
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