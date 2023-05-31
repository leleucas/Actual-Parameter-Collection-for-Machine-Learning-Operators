import csv
import json
import numpy as np
import os
# from sample import *

#>50MB 15.02

def get_input_output_data():
    with open("adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    ratio_list = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input_size"][i]:
            in_size *= dim
        # print(len(arg_data["input_size"][i]))
        assert len(arg_data["input_size"][i]) == 4
        assert len(arg_data["output_size"][i]) == 2
        out_size = arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["output_size"][i][0] * arg_data["output_size"][i][1]
        in_sizes.append(in_size)
        out_sizes.append(out_size)
        ratio_list.append(in_size/out_size)
    return in_sizes, out_sizes, ratio_list

def input_tensor_feature():
    column = 40
    in_sizes, out_sizes, ratio_list = get_input_output_data()
    all_size = len(in_sizes)
    # print(all_size)

    input_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in in_sizes])
    step = int(np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column))
    mem_dict = {}
    label_data = []

    # split the range
    min_val = int(np.min(input_mem_data))
    start = min_val
    for col in range(column):
        end = (int)(start + step)
        label = str(start) + '-' + str(end)
        mem_dict[label] = 0
        label_data.append(label)
        start = end
    # print(mem_dict)
    # count the range data
    for in_data in input_mem_data:
        mem_dict[label_data[int(np.floor((in_data-min_val)/step))]] += 1

    print('max input size ', np.max(input_mem_data))
    print('min input size ', np.min(input_mem_data))
    print('============mem===============\n', mem_dict)
    precent = np.array([value / all_size * 100 for value in mem_dict.values()])
    print('============mem percent============\n', precent)



def output_tensor_feature():
    column = 50
    in_sizes, out_sizes, ratio_list = get_input_output_data()
    all_size = len(out_sizes)
    # print(all_size)

    output_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in out_sizes])
    step = int(np.ceil((np.max(output_mem_data) - np.min(output_mem_data)) / column))
    mem_dict = {}
    label_data = []

    # split the range
    min_val = int(np.min(output_mem_data))
    start = min_val
    for col in range(column):
        end = (int)(start + step)
        label = str(start) + '-' + str(end)
        mem_dict[label] = 0
        label_data.append(label)
        start = end
    # print(mem_dict)
    # count the range data
    for out_data in output_mem_data:
        mem_dict[label_data[int(np.floor((out_data-min_val)/step))]] += 1

    print('max output size ', np.max(output_mem_data))
    print('min output size ', np.min(output_mem_data))
    print('============mem===============\n', mem_dict)
    precent = np.array([value / all_size * 100 for value in mem_dict.values()])
    print('============mem percent============\n', precent)


def inout_ratio_feature():
    column = 100
    in_sizes, out_sizes, ratio_list = get_input_output_data()
    all_size = len(ratio_list)
    # print(all_size)

    # output_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in ratio_list])
    step = int(np.ceil((np.max(ratio_list) - np.min(ratio_list)) / column))
    mem_dict = {}
    label_data = []

    # split the range
    min_val = int(np.min(ratio_list))
    start = min_val
    for col in range(column):
        end = (int)(start + step)
        label = str(start) + '-' + str(end)
        mem_dict[label] = 0
        label_data.append(label)
        start = end
    # print(mem_dict)
    # count the range data
    for ratio in ratio_list:
        mem_dict[label_data[int(np.floor((ratio-min_val)/step))]] += 1

    print('max ratio ', np.max(ratio_list))
    print('min ratio ', np.min(ratio_list))
    print('============mem===============\n', mem_dict)
    precent = np.array([value / all_size * 100 for value in mem_dict.values()])
    print('============mem percent============\n', precent)


if __name__ == '__main__':
    input_tensor_feature()
    output_tensor_feature()
    # inout_ratio_feature()







# column = 15
# in_sizes = get_input_data()

# all_size = len(in_sizes)
# print(all_size)

# input_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in in_sizes])  # KB
# # input_mem_data = np.array(in_sizes)  
# print(np.max(input_mem_data))
# print(np.min(input_mem_data))
# step = int(np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column))

# mem_dict = {}
# label_data = []

# # split the range
# start = int(np.min(input_mem_data))
# for col in range(column):
#     end = (int)(start + step)
#     label = str(start) + '-' + str(end)
#     mem_dict[label] = 0
#     label_data.append(label)
#     start = end
# print(mem_dict)
# # count the range data
# for in_data in input_mem_data:
#     mem_dict[label_data[int(np.floor(in_data/step))]] += 1

# print(mem_dict)
# precent = np.array([value / all_size * 100 for value in mem_dict.values()]) 
# print(precent)
# print(np.min(input_mem_data))
# print(np.max(input_mem_data))

# -------------------------

# import matplotlib.pyplot as plt

# fontsize_tune = 7
# title_size = 10
# colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']

# allrown = 1
# fig, axes = plt.subplots(nrows=allrown, ncols=1, figsize=(11,8*allrown))
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)    #subplots创建多个子图


# ########################## the first row ################################
# rown = 0
# ax0 = axes
# # ax1 = axes[1]
# # ax2 = axes[2]
# # ax01 = axes[rown, 1]
# # ax02 = axes[rown, 2]
# # ax03 = axes[rown, 3]
# # ax04 = axes[rown, 4]

# axis = ax0
# axis.set_ylabel('number', fontsize=10)
# data = list(mem_dict.values())
# names = list(mem_dict.keys())

# axis.bar(names, data, color=colors)
# axis.set_title('Memory size distribution of input data(MB)'
#                , fontsize=title_size)
# axis.tick_params(axis='x', labelsize=fontsize_tune)
# axis.tick_params(axis='y', labelsize=fontsize_tune)
# axis.set_xticks(range(0,len(names),1))
# axis.set_xticklabels(names,rotation=45)


# plt.savefig('adaptive_avg_pool2d.png', dpi=600)