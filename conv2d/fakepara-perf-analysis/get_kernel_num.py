import os
import numpy as np
import pandas as pd
import json

def get_kernel_num(filename):
    kernel_nums = []
    with open(filename, 'r') as f:
        content = f.readline()
        while True:
            cnt = 0
            content = f.readline()
            if 'Disconnected' in content:
                break
            while '***' not in content:
                if '==PROF==' in content:
                    cnt += 1
                content = f.readline()
            kernel_nums.append(cnt)

        return kernel_nums

if __name__ == '__main__':
    kernel_num = get_kernel_num('./data/kernel_numbers1_1.dat')
    print(len(kernel_num), '\n', kernel_num)

        
