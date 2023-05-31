import csv
import json
import os
import ast
import numpy as np
import torch

class SampleConfig(object):
    def __init__(self,
                 args_cases=[],
                 requires_grad=[],
                 backward=False,
                 warm_up_iters=10,
                 performance_iters=1000,
                 timeline_iters=10,
                 save_timeline=False,
                 rtol=1e-04,
                 atol=1e-04,
                 url=None,
                 tags=[]):
        assert len(args_cases) > 0
        self._args_cases = args_cases
        self._requires_grad = requires_grad
        self._backward = backward
        self._warm_up_iters = warm_up_iters
        self._performance_iters = performance_iters
        self._timeline_iters = timeline_iters
        self._save_timeline = save_timeline
        self._rtol = rtol
        self._atol = atol
        self._url = url
        self._tags = tags

    @property
    def args_cases(self):
        return self._args_cases

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def backward(self):
        return self._backward

    @property
    def warm_up_iters(self):
        return self._warm_up_iters

    @property
    def performance_iters(self):
        return self._performance_iters

    @property
    def timeline_iters(self):
        return self._timeline_iters

    @property
    def save_timeline(self):
        return self._save_timeline

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def tags(self):
        return self._tags

    def show_info(self):
        tags = ""
        for tag in self._tags:
            tags = tags + tag.value + " "
        return self._url, tags

    def show(self):
        url, tags = self.show_info()
        print("url:", url)
        print("tags:", tags)


def argIsNone(args):
    if isinstance(args, list):
        assert len(args) == 1
        if args[0] == None :
            return True
        elif isinstance(args[0], str):
            if args[0].lower() == 'none':
                return True
        else:
            return False
    else:
        if args == None:
            return True
        elif isinstance(args, str):
            if args.lower == "none":
                return True
        else:
            return False

def get_sample_config(jsonfile):
    #with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d_dedu.json", "r") as f:
    with open(jsonfile, "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_size"][i], arg_data["output_size"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        url="",  # noqa
        tags=[],
    )

def gen_np_args(input_size_, output_size_):
    input_np = np.random.random(input_size_).astype(np.float32)
    output_size = output_size_

    return [input_np, output_size]



def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).cuda()
    output_size = np_args[1]

    return [input_torch, output_size]



def get_input_data():
    with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_data():
    with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input_size"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_info():
    with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_infos = []
    for i in range(arg_data_length):
        in_info = str(arg_data["input_size"][i][0])
        for dim in range(len(arg_data["input_size"][i])-1):
            in_info += "x" + str(arg_data["input_size"][i][dim+1])
        in_infos.append(in_info)
    return in_infos

def get_vritual_input_data():
    with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input_size"][i]:
            in_size *= dim[0] # List
        in_sizes.append(in_size)
    return in_sizes

def get_vritual_input_info():
    with open("/home/liuhangda/opinsight/DL-Op-Analysis/Pooling/adaptive_avg/adaptive_avg_pool2d_vritual.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_infos = []
    for i in range(arg_data_length):
        in_infos.append("2^"+str(i+1))
    return in_infos
