import csv
import json
import os
from sample import *


column = 10
in_sizes, label_sizes = get_input_data()

all_size = len(in_sizes)

input_mem_data = np.array([(i * 4.0 / 1024 / 1024) for i in in_sizes])  # MB
step = np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column )

mem_dict = {}
label_data = []

# split the range
start = int(np.min(input_mem_data))
for col in range(column):
    end = (int)(start + step)
    label = str(start) + '-' + str(end)
    mem_dict[label] = 0
    label_data.append(label)
    start = end

# count the range data
for in_data in input_mem_data:
    mem_dict[label_data[int(np.floor(in_data/step))]] += 1

precent = np.array([value / all_size * 100 for value in mem_dict.values()]) 
print(precent)
print(np.min(input_mem_data))

# -------------------------

label_num_data = np.array([i for i in label_sizes])  

step = np.ceil((np.max(label_num_data) - np.min(label_num_data)) / column)

label_num_dict = {}
label_data = []

# split the range
start = int(np.min(label_num_data))
for col in range(column):
    end = (int)(start + step)
    label = str(start) + '-' + str(end)
    label_num_dict[label] = 0
    label_data.append(label)
    start = end

# count the range data
for in_data in label_num_data:
    label_num_dict[label_data[int(np.floor(in_data/step))]] += 1

precent = np.array([value / all_size * 100 for value in label_num_dict.values()]) 
print(precent)
# -------------------------

reduce_num_data = np.array([i for i in in_sizes])  //  np.array([i for i in label_sizes]) / 1e3

step = np.ceil((np.max(reduce_num_data) - np.min(reduce_num_data)) / column)

reduce_num_dict = {}
label_data = []

# split the range
start = int(np.min(reduce_num_data))
for col in range(column):
    end = (int)(start + step)
    label = str(start) + '-' + str(end)
    reduce_num_dict[label] = 0
    label_data.append(label)
    start = end


# count the range data
for in_data in reduce_num_data:
    reduce_num_dict[label_data[int(np.floor(in_data/step))]] += 1

precent = np.array([value / all_size * 100 for value in reduce_num_dict.values()]) 
print(precent)

import matplotlib.pyplot as plt

fontsize_tune = 7
title_size = 10
colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']

allrown = 1
fig, axes = plt.subplots(nrows=allrown, ncols=3, figsize=(15,6*allrown))
# plt.xticks(rotation=50)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)    #subplots创建多个子图


########################## the first row ################################
rown = 0
ax0 = axes[0]
ax1 = axes[1]
ax2 = axes[2]
# ax01 = axes[rown, 1]
# ax02 = axes[rown, 2]
# ax03 = axes[rown, 3]
# ax04 = axes[rown, 4]

axis = ax0
axis.set_ylabel('number', fontsize=8)
data = list(mem_dict.values())
names = list(mem_dict.keys())

axis.bar(names, data, color=colors)
axis.set_title('Memory size distribution of input data(MB)'
               , fontsize=title_size)
axis.tick_params(axis='x', labelsize=fontsize_tune)
axis.tick_params(axis='y', labelsize=fontsize_tune)
axis.set_xticks(range(0,len(names),1))
axis.set_xticklabels(names,rotation=45)

axis = ax1
axis.set_ylabel('number', fontsize=8)
data = list(label_num_dict.values())
names = list(label_num_dict.keys())

axis.bar(names, data, color=colors)
axis.set_title('Label size distribution of configuration'
               , fontsize=title_size)
axis.tick_params(axis='x', labelsize=fontsize_tune)
axis.tick_params(axis='y', labelsize=fontsize_tune)
axis.set_xticks(range(0,len(names),1))
axis.set_xticklabels(names,rotation=45)

axis = ax2
axis.set_ylabel('number', fontsize=8)
data = list(reduce_num_dict.values())
names = list(reduce_num_dict.keys())

axis.bar(names, data, color=colors)
axis.set_title('Distribution of number of elements to be reduced(*1e3)'
               , fontsize=title_size)
axis.tick_params(axis='x', labelsize=fontsize_tune)
axis.tick_params(axis='y', labelsize=fontsize_tune)
axis.set_xticks(range(0,len(names),1))
axis.set_xticklabels(names,rotation=45)

plt.savefig('cross_entropy_para.png', dpi=600)