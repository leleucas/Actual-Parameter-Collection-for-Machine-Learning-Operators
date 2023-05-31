import os
import numpy as np
import pandas as pd
import json

shape_out = {"N": [], "C_in": [], "H": [], "W": [], "C_out": [], "kernel_R": [], "kernel_S": [], "strideU": [], "strideV": [], "pad_h": [], "pad_w": [], "algoMin": [], "kernel_nums": [], "count": []}
arg = ["N", "C_in", "H", "W", "C_out", "kernel_R", "kernel_S", "strideU", "strideV", "pad_h", "pad_w", "algoMin", "kernel_nums", "count"]


count_file = pd.read_csv(open("./conv2d_results_count0929.csv"))
count_file_df = pd.DataFrame(count_file)

for it in range(0, 87):
    # print(it)
    if os.path.exists("./kernel_nums/kernel_numbers"+str(it)+".txt"):
        with open("./kernel_nums/kernel_numbers"+str(it)+".txt", 'r') as f:
            byr = f.readline()
            kernel_nums = []
            while True:
                count = 0
                byr = f.readline()
                if 'Disconnected' in byr:
                    break
                while '***' not in byr:
                    if 'large' in byr:
                        count -= 1
                    count += 1
                    byr = f.readline()
                kernel_nums.append(count)
            kernel_nums[0] -= 1

        with open("./实参数据top3000的csv和json/top4349.json", 'r') as f:
            shape_dict_row = json.load(f)
            for i in range(50):
                if it*50+i < 4349:
                    for j in range(len(arg)-2):
                        shape_out[arg[j]].append(shape_dict_row[arg[j]][it*50+i])
                    shape_out[arg[j+1]].append([kernel_nums[i]])
                    shape_out[arg[j+2]].append([int(count_file_df["counts"][it*50+i])])
shape_out_json = json.dumps(shape_out)
with open('./实参数据top3000的csv和json/top4349_count.json', 'w') as f:
    f.write(shape_out_json)
print("数据总数", len(shape_out["N"]))
