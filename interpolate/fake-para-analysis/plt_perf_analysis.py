import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd


rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 28


def print_para(filename):
    data = pd.read_csv(filename, usecols=['l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed', 
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'Time'])
    
    data.to_csv('metric_data.csv', header=True, index=False)


def nearest2d_scale2_perf(filename, ax, colors, fontsize_tune):
    data = pd.read_csv(filename)
    data.head()
    perf = data['perf'].values
    
    yticks = [1, 2, 3, 4]
    xs = [1, 2, 3, 4, 5]
    # yticks = [2, 4, 8, 16, 32]
    # xs = [2, 4, 8, 16, 32]
    cnt = 0
    width = [1, 1, 1, 1, 1]
    for c, k in zip(colors, yticks):
        ys1 = perf[cnt:cnt+5]

        cs = ['tab:brown'] * len(xs)
        ax.bar(xs, ys1, zs=k, zdir='y', width=0.25, color=cs) # alpha = 0.8
        cnt += 5
    
    ax.set_xlabel('case', fontsize=fontsize_tune, labelpad=25) #, labelpad=25
    ax.set_ylabel('memory access band', fontsize=fontsize_tune, labelpad=15) #, labelpad=15
    ax.set_zlabel('memory utilization', fontsize=fontsize_tune, labelpad=10) # labelpad=10


def nearest2d_scale2(filename, ax, colors, fontsize_tune):
    data = pd.read_csv(filename, usecols=[3, 4])
    data.head()
    perf1 = data['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'].values
    perf2 = data['lts__throughput.avg.pct_of_peak_sustained_elapsed'].values
    # print(perf1[99:125])
    # print(perf2[99:125])
    # print(len(perf1))
    yticks = [1, 2, 3, 4, 5]  # 32, 64, 128, 256, 512
    xs1 = [1, 2, 3, 4, 5, 6, 7] # 8, 12, 16, 25, 32, 48, 64
    width = 0.2
    xs2 = [ele+width for ele in xs1]
    print(xs2)
    # yticks = [2, 4, 8, 16, 32]
    # xs = [2, 4, 8, 16, 32]
    cnt = 0
    for k in yticks:
        # if k == 4 or k == 6: continue
        ys1 = perf1[cnt:cnt+7]
        ys2 = perf2[cnt:cnt+7]

        cs = ['tab:olive'] * len(xs1)
        ax.bar(xs1, ys1, zs=k, zdir='y', width=0.2, color=cs) # alpha = 0.8
        cs = ['tab:blue'] * len(xs1)
        ax.bar(xs2, ys2, zs=k, zdir='y', width=0.2, color=cs) # alpha = 0.8
        cnt += 7
    
    ax.set_xlabel('ih=iw', fontsize=fontsize_tune, labelpad=25) #, labelpad=25
    ax.set_ylabel('cin', fontsize=fontsize_tune, labelpad=15) #, labelpad=15
    ax.set_zlabel('L1/L2 cache utilization', fontsize=fontsize_tune, labelpad=10) # labelpad=10


def manage_fig_para(func_name, filename):
    font = { 'size'   : 18}
    plt.rc('font', **font)
    rcParams['font.serif'] = ['Times New Roman']
    fontsize_tune = 18
    fig = plt.figure(figsize=(16, 22)) #figsize=(18, 18)
    ax = fig.add_subplot(projection='3d')
    
    colors = ['r', 'g', 'b', 'y', 'c']
    nearest2d_scale2(filename, ax, colors, fontsize_tune)
    # nearest2d_scale2_perf(filename, ax, colors, fontsize_tune)
    
    ax.tick_params(axis='x', labelsize=fontsize_tune)
    ax.tick_params(axis='y', labelsize=fontsize_tune)
    plt.savefig('interpolate.'+func_name+'.png', format='png', bbox_inches='tight')
    plt.savefig('interpolate.'+func_name+'.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    # print_para('./nearest_metric_data.csv')
    # manage_fig_para('nearest', './plt_data.csv')
    manage_fig_para('nearest', './metric_data.csv')
    # manage_fig_para('bilinear', 'bilinear.xlsx')