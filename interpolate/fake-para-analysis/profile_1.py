import argparse
import torch
from torch.nn import functional
import numpy as np
import time
import logging
import json

from collections import defaultdict, OrderedDict


torch.cuda.set_device(1)

#logging.basicConfig(filename="1381.log",level=logging.DEBUG)
#torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False


def get_sample_config():
    with open("linear.json", "r") as f:
        arg_data = json.load(f)
    return arg_data


def gen_np_args(x1_, x2_):
    x1 = np.random.random(x1_).astype(np.float32)
    x2 = np.random.random(x2_).astype(np.float32)
    return [x1, x2]

def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    return [x1, x2]


def virtual_arg_profile():
    inputparas = defaultdict(list)
    pow = 28 #2^1 - 2^28

    for i in range(1,2):
        batch = 2**(i)
        for j in range(1):
            in_features = 2**(j+1)
            for k in range(20):
                out_features = 2**(k+1)

                inputshape = [batch, in_features]
                weightshape = [out_features, in_features]
                print(batch, in_features, out_features)

                inputparas["input"].append(inputshape)
                inputparas["weight"].append(weightshape)

                np_arg = gen_np_args(inputshape, weightshape)
                torch_arg = args_adaptor(np_arg)
                
                import time
                out = functional.linear(input=torch_arg[0],weight=torch_arg[1]).cuda() 
                torch.cuda.synchronize()
                time1 = time.time()

                out = functional.linear(input=torch_arg[0],weight=torch_arg[1]).cuda()
                torch.cuda.synchronize()
                time = time.time() - time1
                print(time)
    json_str = json.dumps(inputparas, indent=4)
    with open('top3000.json', 'w') as json_file:
        json_file.write(json_str)


def profile(device):
    samples = get_sample_config()
    samples_num = len(samples)
    # print("samples ", samples_num)
    startTime = time.time()
    #for i in range(samples_num):
    for i in range(1):
        print(i, "#" * 20)
        case_args = samples[i]
        input_tensor = torch.from_numpy(np.ones(case_args['Input'][0]))
        weight_tensor = torch.from_numpy(np.ones(case_args['Weight'][0]))
        bias_tensor = False
        #if case_args["Bias"] != []:
        #    if case_args['Bias'][0] != [False]:
        #        bias_tensor = torch.from_numpy(np.ones(case_args['Bias'][0])#.astype(np.float32)).cuda()

        if type(bias_tensor) == type(False):
            out = functional.linear(input=input_tensor,weight=weight_tensor).cuda()
        else:
            out = functional.linear(input=input_tensor,weight=weight_tensor,bias=bias_tensor).cuda()
        logging.debug("finish " + str(i) + " time " + str(time.time()-startTime))



def main():
    # cuda settings
    use_cuda = torch.cuda.is_available()
    assert use_cuda == True, "cuda environment is not ready"
    device = torch.device("cuda")
    # profile(device)
    virtual_arg_profile()

if __name__ == '__main__':
    main()
