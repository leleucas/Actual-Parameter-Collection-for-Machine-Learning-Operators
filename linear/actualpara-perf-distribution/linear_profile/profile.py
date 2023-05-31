import argparse
import torch
from torch.nn import functional
import numpy as np
import time
import logging
import json
logging.basicConfig(filename="1381.log",level=logging.DEBUG)
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False


def get_sample_config():
    with open("linear.json", "r") as f:
        arg_data = json.load(f)
    return arg_data

def profile(device):
    samples = get_sample_config()
    samples_num = len(samples)
    # print("samples ", samples_num)
    startTime = time.time()
    for i in range(samples_num):
    #for i in range(20):
        print(i, "#" * 20)
        case_args = samples[i]
        input_tensor = torch.from_numpy(np.ones(case_args['Input'][0]).astype(np.float32)).cuda()
        weight_tensor = torch.from_numpy(np.ones(case_args['Weight'][0]).astype(np.float32)).cuda()
        bias_tensor = False
        if case_args["Bias"] != []:
            if case_args['Bias'][0] != [False]:
                bias_tensor = torch.from_numpy(np.ones(case_args['Bias'][0]).astype(np.float32)).cuda()

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
    profile(device)
    # virtual_arg_profile()

if __name__ == '__main__':
    main()
