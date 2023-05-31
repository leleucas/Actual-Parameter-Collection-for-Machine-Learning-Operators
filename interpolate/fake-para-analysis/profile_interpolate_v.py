import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import argparse
import torch
from torch.nn import functional
import numpy as np
import time
import logging
import json

from collections import defaultdict, OrderedDict

# torch.cuda.set_device(1)

import argparse
import torch
from torch.nn import functional
import numpy as np
import time
import logging
import json
logging.basicConfig(filename="5999.log",level=logging.DEBUG)
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False

def gen_np_args(x1_):#, x2_):
    x1 = np.random.random(x1_).astype(np.float32)
    return [x1]#[x1, x2]


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    return [x1]#[x1, x2]


def virtual_arg_profile():
    inputparas = defaultdict(list)
    pow = 28 #2^1 - 2^28
    c_list = [32, 48, 64, 96, 128, 256, 512]
    k_list = [8, 12, 16, 25, 32, 48, 64]

    for i in range(6,7):
        mini_batch = 2**(i)
        for j in c_list:
            channels = j
            for k in k_list:
                optionaldepth = width = k

                inputshape = [mini_batch, channels, optionaldepth, width]
                print(mini_batch, channels, optionaldepth, width)

                inputparas["input_tensor"].append(inputshape)

                np_arg = gen_np_args(inputshape)
                torch_arg = args_adaptor(np_arg)
                
                out = functional.interpolate(torch_arg[0],size=None,scale_factor=2,mode='nearest') #,align_corners=False
    
    json_str = json.dumps(inputparas, indent=4)
    with open('nearest.json', 'w') as json_file:
        json_file.write(json_str)


def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    # profile(device)
    virtual_arg_profile()


if __name__ == '__main__':
    main()