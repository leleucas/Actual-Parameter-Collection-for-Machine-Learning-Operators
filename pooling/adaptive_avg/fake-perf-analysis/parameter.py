import csv
import json
import os
from sample import *


column = 15
in_sizes = get_input_data()

all_size = len(in_sizes)
print(all_size)

input_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in in_sizes])  # KB
print(np.max(input_mem_data))
print(np.min(input_mem_data))
step = int(np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column))

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
print(mem_dict)
for in_data in input_mem_data:
    mem_dict[label_data[int(np.floor(in_data/step))]] += 1

print(mem_dict)
precent = np.array([value / all_size * 100 for value in mem_dict.values()]) 
print(precent)
print(np.min(input_mem_data))
print(np.max(input_mem_data))

# -------------------------

import matplotlib.pyplot as plt

fontsize_tune = 7
title_size = 10
colors = ['tomato', 'lightskyblue', 'goldenrod', 'green', 'y']

allrown = 1
fig, axes = plt.subplots(nrows=allrown, ncols=1, figsize=(11,8*allrown))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)    #subplots创建多个子图


########################## the first row ################################
rown = 0
ax0 = axes
# ax1 = axes[1]
# ax2 = axes[2]
# ax01 = axes[rown, 1]
# ax02 = axes[rown, 2]
# ax03 = axes[rown, 3]
# ax04 = axes[rown, 4]

axis = ax0
axis.set_ylabel('number', fontsize=10)
data = list(mem_dict.values())
names = list(mem_dict.keys())

axis.bar(names, data, color=colors)
axis.set_title('Memory size distribution of input data(MB)'
               , fontsize=title_size)
axis.tick_params(axis='x', labelsize=fontsize_tune)
axis.tick_params(axis='y', labelsize=fontsize_tune)
axis.set_xticks(range(0,len(names),1))
axis.set_xticklabels(names,rotation=45)


plt.savefig('adaptive_avg_pool2d.png', dpi=600)