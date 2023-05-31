import os
import numpy as np
import pandas as pd
import collections
import json
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams


def print_para(filename):
    data = pd.read_csv(filename, usecols=['input_size', 'output_size', 
        'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed', 
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 
        'Bandwidth',
        'bw_ratio',
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'Time'])
    
    data.to_csv('metric_data.csv', header=True, index=False)


def plt_factors(filename):
    data = pd.read_csv(filename)
    # in_sizes = data['input_size']
    # out_sizes = data['output_size']
    l1_util = data['l1tex__throughput.avg.pct_of_peak_sustained_elapsed']
    l2_util = data['lts__throughput.avg.pct_of_peak_sustained_elapsed']

    plt_metrics = ['lts__throughput.avg.pct_of_peak_sustained_elapsed',
                    'l1tex__throughput.avg.pct_of_peak_sustained_elapsed']

    plt_size = [[[100, "64x32x64x64"], [140, "64x128x64x64"], [180, "64x512x64x64"], [220, "64x2048x64x64"]]
                , [[110, "64x32x97x97"], [150, "64x128x97x97"], [190, "64x512x97x97"], [230, "64x2048x97x97"]]]
    # out_sizes = ['1x1', '2x2', '3x3', '4x4', '5x5', '6x6', '7x7', '8x8', '16x16', '32x32']
    out_sizes = ['1', '2', '3', '4', '5', '6', '7', '8', '16', '32']
    plt_rect = []
    # DEDU
    # in_size_utilzation_dict = collections.OrderedDict()
    for i in range(len(plt_size)):
        in_size_utilzation_dict = collections.OrderedDict()
        for ele in plt_size[i]:
            start_idx = ele[0]
            in_size = ele[1]
            l1_val_list = []
            l2_val_list = []
            for step_idx in range(1, len(out_sizes)):
                # print(step_idx)
                idx = start_idx + step_idx
                # print(idx)
                l1_val_list.append(l1_util[idx])
                l2_val_list.append(l2_util[idx])
            in_size_utilzation_dict[in_size] = [l1_val_list, l2_val_list]
        plt_rect.append(collections.OrderedDict(sorted(in_size_utilzation_dict.items())))

    print('-----------------------------')


    fig, axs = plt.subplots(2, 4, figsize=(12, 10), sharey='row', layout='constrained')  # sharex='col'

    font = { 'size'   : 18}
    plt.rc('font', **font)
    rcParams['font.serif'] = ['Times New Roman']
    fontsize_tune = 15
    colors = ['tab:orange', 'tab:blue']    #'tab:brown', 'tab:olive', 'tab:red', 'tab:green'
    width = 0.25
    x = np.arange(0, len(out_sizes)-1)
    for i in range(len(plt_size)):
        data_dict = plt_rect[i]
        for j in range(len(plt_size[i])):
            
            in_size = plt_size[i][j][1]
            ax = axs[i, j]
            l1_val_list = data_dict[in_size][0]
            l2_val_list = data_dict[in_size][1]
            
            b0 = ax.bar(x, l1_val_list, width, label='l1 utilization', color=colors[0])
            b1 = ax.bar(x+width, l2_val_list, width, label='l2 utilization', color=colors[1])
            # ax.set_ylabel(plt_metrics[i])
            ax.set_xlabel('oh=ow\ninput_size='+in_size, size=15)
            ax.tick_params(axis='x', labelsize=fontsize_tune)
            ax.tick_params(axis='y', labelsize=fontsize_tune)
            ax.set_xticks(range(0,len(out_sizes)-1,1))
            # ax.set_yticks(range(0,101,20))
            ax.set_xticklabels(out_sizes[1:]) #,rotation=50
    legend_value = ["L1 Cache Utilization", "L2 Cache Utilization"] 
    leg = fig.legend([b0, b1], legend_value, bbox_to_anchor=[0.27, 0.993], prop={'size':13})
    axs[0,0].set_ylabel('Cache Utilization(%)', size=15)
    axs[1,0].set_ylabel('Cache Utilization(%)', size=15)
    plt.savefig('pooling.perf.analysis.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('pooling.perf.analysis.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    # plt_factors('./data/pooling_para_metric64.csv')
    print_para('./data/pooling_para_metric64.csv')
