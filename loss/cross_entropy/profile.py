import argparse
import torch
from torch.nn import functional
import numpy as np
import time
from sample import *

def cross_entropy(input, target, weight_,
                size_average_, ignore_index_, reduce_, reduction_, label_smoothing_):
    output = functional.cross_entropy(input, target, weight=weight_,
            size_average=size_average_, ignore_index=ignore_index_, 
            reduce=reduce_, reduction=reduction_, label_smoothing=label_smoothing_)
    return output

def layout_profile():
    samples = get_sample_config()
    print(samples._args_cases[0])
    frist_args = samples._args_cases[0]
    in_shape = [2*512*512,150]
    target_shape = [2*512*512]
    np_arg = gen_np_args(in_shape, target_shape, frist_args[2], frist_args[3],
                         frist_args[4], frist_args[5], frist_args[6], frist_args[7])
    torch_arg = args_adaptor(np_arg)
    print(torch_arg)
    out = cross_entropy(torch_arg[0], torch_arg[1], torch_arg[2], torch_arg[3],
                  torch_arg[4], torch_arg[5], torch_arg[6], torch_arg[7])
    print(out)


def profile(device):   
    samples = get_sample_config()
    print(samples._args_cases[0])
    samples_num = len(samples._args_cases)
    for i in range(samples_num):
        case_args = samples._args_cases[i]
        np_arg = gen_np_args(case_args[0], case_args[1], case_args[2], case_args[3],
                            case_args[4], case_args[5], case_args[6], case_args[7])
    torch_arg = args_adaptor(np_arg)
    print(torch_arg)
    out = cross_entropy(torch_arg[0], torch_arg[1], torch_arg[2], torch_arg[3],
                  torch_arg[4], torch_arg[5], torch_arg[6], torch_arg[7])


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    profile(device)
    layout_profile()

if __name__ == '__main__':
    main()
