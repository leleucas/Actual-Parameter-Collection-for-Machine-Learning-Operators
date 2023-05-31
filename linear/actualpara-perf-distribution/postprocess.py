# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
from roofline import roofline
import json
import csv
import sys

dfs={}

shape_out = {"N": [], "cin": [], "cout": []}
arg = ["N", "cin", "cout"]

def append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric_info, cnt, count_perf, max_perf, min_perf, aihbm_list, rangecnt):
    count_perf[rangecnt] += 1
    max_perf[rangecnt] = max(max_perf[rangecnt], float(aihbm_list[cnt]))
    min_perf[rangecnt] = min(min_perf[rangecnt], float(aihbm_list[cnt]))
    metric_info['ai'].append(aihbm_list[cnt])
    metric_info['perf'].append(perf_list[cnt])
    metric_info['tc_perf'].append(tc_list[cnt])
    metric_info['time'].append(time_list[cnt])
    for j in range(len(arg)):
        metric_info[arg[j]].append(kernel_data_dict[arg[j]][cnt])

def write_csv(metric_info, range):
    datacsv = pd.DataFrame(metric_info)
    savepath = './data/'
    if not os.path.exists(savepath): os.mkdir(savepath)
    datacsv.to_csv(savepath+'/perfrange'+range+'.csv', index=False, header=True)

def print_metricinfo(dfmetric, kerneljson):
    header_list = arg + ['time', 'ai', 'tc_perf', 'perf']
    metricabove70_info = {i:[] for i in header_list}
    metric6070_info = {i:[] for i in header_list}
    metric5060_info = {i:[] for i in header_list}
    metric4050_info = {i:[] for i in header_list}
    metric3040_info = {i:[] for i in header_list}
    metric2030_info = {i:[] for i in header_list}
    metricunder20_info = {i:[] for i in header_list}

    kernel_data_dict = {}
    with open(kerneljson, 'r') as f:
        kernel_data_dict = json.load(f)

    count_perf = [0 for i in range(7)]
    max_perf = [0.0 for i in range(7)]
    min_perf = [1e6 for i in range(7)]
    perf_list = dfmetric['GFLOP/s'].values
    tc_list = dfmetric['TC GFLOP/s'].values
    time_list = dfmetric['Time'].values
    aihbm_list = dfmetric['AI HBM'].values
    max_perf_val = 0.0

    peak_perf = 19.5 * 1000
    for i in range(len(perf_list)):
        if perf_list[i] > max_perf_val:
            max_perf_val = perf_list[i]
        if abs(peak_perf - perf_list[i])/peak_perf <= 0.3:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metricabove70_info, i, count_perf, max_perf, min_perf, aihbm_list, 0)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.3 and abs(peak_perf - perf_list[i])/peak_perf <= 0.4:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric6070_info, i, count_perf, max_perf, min_perf, aihbm_list, 1)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.4 and abs(peak_perf - perf_list[i])/peak_perf <= 0.5:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric5060_info, i, count_perf, max_perf, min_perf, aihbm_list, 2)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.5 and abs(peak_perf - perf_list[i])/peak_perf <= 0.6:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric4050_info, i, count_perf, max_perf, min_perf, aihbm_list, 3)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.6 and abs(peak_perf - perf_list[i])/peak_perf <= 0.7:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric3040_info, i, count_perf, max_perf, min_perf, aihbm_list, 4)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.7 and abs(peak_perf - perf_list[i])/peak_perf <= 0.8:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metric2030_info, i, count_perf, max_perf, min_perf, aihbm_list, 5)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.8:
            append_kernelinfo(perf_list, tc_list, time_list, kernel_data_dict, metricunder20_info, i, count_perf, max_perf, min_perf, aihbm_list, 6)
    
    write_csv(metricabove70_info, 'above70')
    write_csv(metric6070_info, '60-70')
    write_csv(metric5060_info, '50-60')
    write_csv(metric4050_info, '40-50')
    write_csv(metric3040_info, '30-40')
    write_csv(metric2030_info, '20-30')
    write_csv(metricunder20_info, 'under20')

    total_num = len(perf_list)
    stringrange = ['rangeabove70', 'range60-70', 'range50-60', 'range40-50', 'range30-40', 'range20-30', 'rangeunder20']
    print('total cases: ', total_num)
    for i in range(7):
        print(stringrange[i], ' ', max_perf_val, ' ', max_perf_val/peak_perf, ' ', count_perf[i], ' ', count_perf[i]/total_num, ' ', max_perf[i], ' ',min_perf[i])


with open("./linear_para.json", 'r') as f:
    shape_out = json.load(f)
file = 'nsight1381.csv'

tag, ext = os.path.splitext(os.path.basename(file))
dfs[tag]=pd.DataFrame()
with open(file,'r') as f:
    df = pd.read_csv(file)
    df_group=pd.DataFrame()


    dftmp=pd.DataFrame(df, columns=['ID', 'Kernel Name','Metric Name', 'Metric Value'])
    # filter
    def fliter(x):
        fliterList = ["ampere_sgemm", "sgemm_largek", "gemmSN_TN_kernel", "gemv2T_kernel", "dot_kernel"]
        pointList = [0, 0, 0, 1, 2]
        for i in (0,1,2,3,4):
            if fliterList[i] in x:
                return pointList[i]+1
        else:
            return -1
    dftmp["obj"] = dftmp['Kernel Name'].apply(fliter)


    dft = dftmp[dftmp['obj'] != -1]
    cur_num = dft.shape[0] // 21
    print("cur_num:", cur_num)

    dft = dft[['ID', 'Kernel Name', 'Metric Name', 'Metric Value', 'obj']]
    dft['Metric Value'] = pd.to_numeric(dft['Metric Value'].str.replace(r',',''))
    dfmetric=pd.pivot_table(dft, index=['ID'], columns=['Metric Name'], values=['Metric Value'])
    dfmetric.to_csv("a.csv")
    fa = open("a.csv")
    csv_reader = csv.reader(fa)

    count = 0
    with open("b.csv", 'w') as f:
        csv_writer = csv.writer(f)
        for line in csv_reader:
            if count != 0 and count != 2:
                csv_writer.writerow(line)
            count += 1
        f.close
    count -= 2
    fb = open("b.csv")
    dfmetric = pd.read_csv(fb)

    dfmetric['Time']=dfmetric['sm__cycles_elapsed.avg'] \
                    / (dfmetric['sm__cycles_elapsed.avg.per_second'] )

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
        'dram__bytes.sum',
        'lts__t_bytes.sum',
        'l1tex__t_bytes.sum']

    df_dict = {i: []
    for i in df_list
        }
    cur_case = 0
    cur_line = 0
    while cur_case < cur_num:
        kernel_num = 1
        for j in range(len(df_list)):
            curnum = 0.0
            for i in range(cur_line, cur_line+kernel_num):
                curnum += float(dfmetric[df_list[j]][i])
            df_dict[df_list[j]].append(curnum)
        cur_line += kernel_num
        cur_case += 1

    header = df_dict.keys()
    rows=pd.DataFrame(df_dict).to_dict('records')
    
    with open('deal.csv', 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for data in rows:
            f.write(",".join(str(data[h]) for h in header))
            f.write('\n')
    fd = open("deal.csv")
    dfmetric = pd.read_csv(fd)
    fa.close()
    fb.close()
    fd.close()
    os.remove("a.csv")
    os.remove("b.csv")
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
    para_csv = pd.DataFrame(shape_out)
    para_metric_csv = pd.concat([para_csv, dfmetric], axis=1)
    para_metric_csv.to_csv('./data/linear_para_metric.csv', index=False, header=True)
    dfmetric.to_csv('pd_'+tag+'.csv')
    dfs[tag]=dfmetric

    print_metricinfo(dfmetric, 'linear_para.json')


tags=dfs.keys()
flags=['HBM'] #'HBM','L2','L1' or 'all'
for tag in tags:
    for flag in flags:
        dfm=dfs[tag]
        LABELS = [str(i) for i in range(len(dfm.index.tolist()))]
        AIL1   = dfm['AI L1'].tolist()
        AIL2   = dfm['AI L2'].tolist()
        AIHBM  = dfm['AI HBM'].tolist()
        FLOPS  = dfm['GFLOP/s'].tolist()
        pointType = dft['obj'].tolist()[::21]

        roofline(tag, FLOPS, AIHBM, AIL2, AIL1, LABELS, flag, cur_num, pointType)