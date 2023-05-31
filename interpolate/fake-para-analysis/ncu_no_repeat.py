# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
import collections
import json
import csv
datadir='.'

dfs = {}


NUM_Metrics = 25

file = 'nearest_nsightdat_processed.csv'
tag, ext = os.path.splitext(os.path.basename(file))
dfs[tag]=pd.DataFrame()
with open(file,'r') as f:
    df = pd.read_csv(file)
    dft = pd.DataFrame(df, columns=['ID', 'Kernel Name', 'Metric Name', 'Metric Value'])
    dft['Metric Value'] = pd.to_numeric(dft['Metric Value'].str.replace(r',', ''))
    dfmetric=pd.pivot_table(dft, index=['ID'], columns=['Metric Name'], values=['Metric Value'])
    dfmetric.to_csv("tmp.csv")
    kernel_name = dft['Kernel Name']
    kernel_name_dedu = []
    for k in range(0,  len(kernel_name), NUM_Metrics):
        kernel_name_dedu.append(kernel_name[k])
    df_kernel_name_dedu = pd.DataFrame(kernel_name_dedu, columns=['Kernel Name'])


    csv_reader = csv.reader(open("tmp.csv"))
    os.remove("tmp.csv")
    count = 0
    with open("nsight_result.csv", 'w') as f:
        csv_writer = csv.writer(f)
        for line in csv_reader:
            if count != 0 and count != 2:
                csv_writer.writerow(line)
            count += 1
        f.close
    count -= 2
    
    dfmetric = pd.read_csv(open("nsight_result.csv"))


    dfmetric['Time']=dfmetric['sm__cycles_elapsed.avg'] \
                    / (dfmetric['sm__cycles_elapsed.avg.per_second'] )
    dfmetric['Kernel Name'] = df_kernel_name_dedu['Kernel Name']
    print('numofkernelname ', len(df_kernel_name_dedu['Kernel Name']))

    df_list = ['Time', 'sm__sass_thread_inst_executed_op_dfma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_dmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_dadd_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_fadd_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hfma_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hmul_pred_on.sum', 
         'sm__sass_thread_inst_executed_op_hadd_pred_on.sum', 
         'sm__inst_executed_pipe_tensor.sum',
         'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
         'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
         'lts__throughput.avg.pct_of_peak_sustained_elapsed',
         'dram__bytes.sum',
         'lts__t_bytes.sum',
         'l1tex__t_bytes.sum',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed',
        'Achieved Occupancy', 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
        'Registers Per Thread', 'Threads']

    name_list = []

    df_dict = {i: []
    for i in df_list
        }
    
    n2_cnt = 0
    n3_cnt = 0
    b2_cnt = 0
    c2_cnt = 0
    cur_line = 0
    kernel_keys = ["nearest2d", "nearest3d", "bilinear2d", "bicubic2d"]
    k_start = False
    for index, kernel in dfmetric.iterrows():
        for kernel_key in kernel_keys:
            if kernel_key in kernel['Kernel Name'].lower():
                name_list.append(kernel_key)
                if kernel_key == 'nearest2d':
                    n2_cnt += 1
                elif kernel_key == 'nearest3d':
                    n3_cnt += 1
                elif kernel_key == 'bilinear2d':
                    b2_cnt += 1
                elif kernel_key == 'bicubic2d':
                    c2_cnt += 1

                for j in range(len(df_list)):
                    curnum = 0.0
                    for i in range(cur_line, cur_line+1):
                        curnum += float(dfmetric[df_list[j]][i])
                    df_dict[df_list[j]].append(curnum)
        cur_line += 1
    NUM_DEDU = 49
    assert len(df_dict['Time']) == NUM_DEDU
    print('len of name list: ', len(name_list), n2_cnt, n3_cnt, b2_cnt, c2_cnt)
    
    header = df_dict.keys()
    rows=pd.DataFrame(df_dict).to_dict('records')
    
    with open('deal.csv', 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for data in rows:
            f.write(",".join(str(data[h]) for h in header))
            f.write('\n')

    dfmetric = pd.read_csv(open("deal.csv"))
    os.remove("deal.csv")
    for i in df_list:
        dfmetric[i] = pd.to_numeric(dfmetric[i])


    dfmetric['CC FLOPs']= 2 * dfmetric['sm__sass_thread_inst_executed_op_dfma_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_dmul_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_dadd_pred_on.sum'] \
                        + 2 * dfmetric['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'] \
                        + 2 * dfmetric['sm__sass_thread_inst_executed_op_hfma_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_hmul_pred_on.sum'] \
                        + dfmetric['sm__sass_thread_inst_executed_op_hadd_pred_on.sum'] 

    dfmetric['TC FLOPs']= 512 * dfmetric['sm__inst_executed_pipe_tensor.sum']
    dfmetric['all FLOPs']= dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']
    
    dfmetric['AI HBM'] = dfmetric['all FLOPs'].div(dfmetric['dram__bytes.sum'])
    dfmetric['AI L2'] = dfmetric['all FLOPs'].div(dfmetric['lts__t_bytes.sum'])
    dfmetric['AI L1'] = dfmetric['all FLOPs'].div(dfmetric['l1tex__t_bytes.sum'])

    dfmetric['GFLOP/s'] = dfmetric['all FLOPs']/ dfmetric['Time'] /1024/1024/1024
    dfmetric['TC GFLOP/s'] = dfmetric['TC FLOPs']/ dfmetric['Time'] /1024/1024/1024

    dfmetric.to_csv('nearest_metric_data.csv', index=False, header=True)