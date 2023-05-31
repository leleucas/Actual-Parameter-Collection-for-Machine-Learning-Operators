import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.rc('font',family='Times New Roman') 
from pylab import xticks, yticks, np
import numpy as np
import pandas as pd
import os
from math import log, exp
from matplotlib import rcParams
rcParams['font.serif'] = ['Times New Roman']
rcParams['figure.constrained_layout.use'] = True

peak_perf = 19500

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


def plt_conv2d_perffactor_withfakepara(batch_size, kernel_size, metric, metric_name, res_file):
    dir = './data/' + str(batch_size) + '-' + str(kernel_size) + '/'
    perf_bands = ['rangeunder20', 'range20-30', 'range30-40', 'range40-50', 'range50-60', 'range60-70', 'rangeabove70']
    labels = ['Under20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%', 'Above70%']
    data = []
    for ele in perf_bands:
        metric_data = pd.read_csv(dir+'perf'+ele+'.csv')
        vals = pd.to_numeric(metric_data[metric].values)
        data.append(vals)
    
    fig, ax = plt.subplots(1, 1)
    ax.boxplot(data)
    # ax.violinplot(data)
    ax.yaxis.grid(True)
    ylabels = [str(i) for i in range(0, 160, 20)]
    ax.set_yticks([i for i in range(0, 160, 20)], labels=ylabels, size=18)
    ax.set_xticks([y + 1 for y in range(len(data))],
                labels=labels, rotation=45, size=18)
    ax.set_xlabel('computational performance bands', size=20)
    ax.set_ylabel(metric_name + " (%)", size=18)
    # ax.set_xticks(rotation=45)

    print('====================='+metric_name+'==========================')
    for i in data:
        if len(i) == 0:
            print('null', 'null')
            continue
        print(np.median(i),np.mean(i))

    # plt.savefig(res_file+'.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig(res_file+'.png', format='png', bbox_inches='tight')


# GFLOP/s
# Achieved Occupancy
def plt_conv2d_perf_withfakepara(filename, metric, res_file):
    data = pd.read_csv(filename)
    n = data['input_channel'].values
    ih = data['image_size_H'].values
    gflops = pd.to_numeric(data[metric].values)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xticks_labels = ['{}'.format(4 ** i) for i in range(0,7)]
    xticks(np.linspace(0,6,7,endpoint=True), xticks_labels, fontsize=8)
    yticks_labels = ['{}'.format(4 ** i) for i in range(0,7)]
    yticks(np.linspace(0,6,7,endpoint=True),yticks_labels, fontsize=8)
    # zticks_labels = ['{}'.format(10 ** i) for i in range(0,6)]
    # ax.set_zticks(np.linspace(0,5,6,endpoint=True), zticks_labels, fontsize=8)
    # ax.set_zlim(0, 5)
    ax.set_zlim(0, 15)
    ax.tick_params(labelsize=11)

    ax.set_xlabel('cin = cout', fontsize=13)
    ax.set_ylabel('ih = iw', fontsize=13)
    ax.set_zlabel('TFLOPS', fontsize=10)
    

    for i in range(len(n)):
        if abs(gflops[i] - peak_perf) / peak_perf <= 0.3:
            color = colors[0]
        elif 0.3 < abs(gflops[i] - peak_perf) / peak_perf <= 0.4:
            color = colors[1]
        elif 0.4 < abs(gflops[i] - peak_perf) / peak_perf <= 0.5:
            color = colors[2]
        elif 0.5 < abs(gflops[i] - peak_perf) / peak_perf <= 0.6:
            color = colors[3]
        elif 0.6 < abs(gflops[i] - peak_perf) / peak_perf <= 0.7:
            color = colors[4]
        elif 0.7 < abs(gflops[i] - peak_perf) / peak_perf <= 0.8:
            color = colors[5]
        else:
            color = colors[6]
        # ax.scatter(log(int(n[i]), 10), int(ih[i]), gflops[i], c=color, s=5) # linear
        ax.scatter(log(int(n[i]), 4), log(int(ih[i]), 4), gflops[i]/1000, c=color, s=8) # log

    plt.savefig(res_file+'.pdf', format='pdf') #, bbox_inches='tight'
    plt.savefig(res_file+'.png', format='png', bbox_inches='tight')


def print_metricinfo():
    # batchsize = [16, 32, 64, 128]
    batchsize = [64]
    kernelsize = [1, 3]
    # perf_files = ['perfrange20-30.csv', 'perfrange30-40.csv', 'perfrange40-50.csv',\
    #     'perfrange50-60.csv', 'perfrange60-70.csv', 'perfrangeabove70.csv', 'perfrangeunder20.csv']
    print(batchsize, kernelsize)
    for j in kernelsize:
        for i in batchsize:
            print('=========batchsize=',i,'=======kernelsize=',j,'===========')
            for file in os.listdir('./data/'+str(i)+'-'+str(j)):
                print('*****file ', file)
                data_pd = pd.read_csv('./data/'+str(i)+'-'+str(j)+'/'+file)
                print('=================='+os.path.splitext(file)[0]+'=====================')
                print('perf','occupancy','sm','l1','l2','sharedmem')
                perf_list = pd.to_numeric(data_pd['perf'].values)
                occupancy_list = pd.to_numeric(data_pd['Achieved Occupancy'].values)
                sm_throughput_list = pd.to_numeric(data_pd['sm__throughput.avg.pct_of_peak_sustained_elapsed'].values)
                l1_list = pd.to_numeric(data_pd['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'].values)
                l2_list = pd.to_numeric(data_pd['lts__throughput.avg.pct_of_peak_sustained_elapsed'].values)
                sharedmem_list = pd.to_numeric(data_pd['l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed'].values)
                
                
                sharedmem_bankconflict_list = data_pd['l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum'].values
                cin_list = data_pd['input_channel'].values
                ih_list = data_pd['image_size_H'].values
                k_list = data_pd['kernel_sizeR'].values

                for w in range(len(k_list)):
                    print(cin_list[w], ih_list[w], k_list[w], sharedmem_bankconflict_list[w])

                
                # print('\n')
                # if len(perf_list) != 0:
                #     print(np.median(perf_list), np.median(occupancy_list), np.median(sm_throughput_list),
                #                         np.median(l1_list), np.median(l2_list), np.median(sharedmem_list))
                #     print(np.mean(perf_list), np.mean(occupancy_list), np.mean(sm_throughput_list),
                #                         np.mean(l1_list), np.mean(l2_list), np.mean(sharedmem_list))
                # print('\n')

                # for idx in range(len(perf_list)):
                #     print(perf_list[idx], occupancy_list[idx], sm_throughput_list[idx], l1_list[idx], l2_list[idx], sharedmem_list[idx])
                # print('\n')
            print('\n\n')


if __name__ == '__main__':
    print_metricinfo()

    batchsize = 64
    kernelsize = 1
    # metric = 'Achieved Occupancy'
    # metric_name = 'Occupancy'
    metric = 'sm__throughput.avg.pct_of_peak_sustained_elapsed'
    metric_name = 'SM Utilization'
    plt_conv2d_perffactor_withfakepara(batchsize, kernelsize, metric, metric_name, 
                        'conv2d.gemm.'+metric_name+str(batchsize)+'_'+str(kernelsize))
                    


    # batchsize = 64
    # kernelsize = 1
    # metric = 'GFLOP/s'
    # metric_name = 'perf'
    # # metric = 'Achieved Occupancy'
    # # metric_name = 'occupancy'
    # plt_conv2d_perf_withfakepara('./data/conv2d_para_metric'+str(batchsize)+'_'+str(kernelsize)+'.csv', 
    #                                metric, 'conv2d.gemm.'+metric_name+str(batchsize)+'_'+str(kernelsize))

    # batchsize = 64
    # kernelsize = 3
    # metric = 'GFLOP/s'
    # metric_name = 'perf'
    # # metric = 'Achieved Occupancy'
    # # metric_name = 'occupancy'
    # plt_conv2d_perf_withfakepara('./data/conv2d_para_metric'+str(batchsize)+'_'+str(kernelsize)+'.csv', 
    #                                metric, 'conv2d.wino.'+metric_name+str(batchsize)+'_'+str(kernelsize))