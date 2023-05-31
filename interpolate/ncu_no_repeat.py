# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
import collections
import json
import csv
datadir='.'

dfs = {}

arg = ['N', 'C', 'H', 'W']

repeat_number = pd.read_csv(open("repeat_number.csv"))
ALL_NUMBER = np.sum(repeat_number.values)
NUM_Metrics = 21
NUM_DEDU = len(repeat_number.values)

def get_input_data():
    with open("interpolate_all_top5740_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_input_info(idx_list):
    n_dict = {'n':[], 'cnt':[]}
    c_dict = {'c':[], 'cnt':[]}
    h_dict = {'h':[], 'cnt':[]}
    w_dict = {'w':[], 'cnt':[]}
    total_cnt = 0
    select_cnt = 0
    with open("interpolate_all_top5740_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    for i in range(arg_data_length):
        if i not in idx_list:
            continue
        if len(arg_data["x1"][i]) != 4:
            total_cnt = total_cnt + 1
            continue
        select_cnt = select_cnt + 1
        n = arg_data["x1"][i][0]
        c = arg_data["x1"][i][1]
        h = arg_data["x1"][i][2]
        w = arg_data["x1"][i][3]
        if n not in n_dict['n']:
            n_dict['n'].append(n)
            n_dict['cnt'].append(1)
        else:
            idx = n_dict['n'].index(n)
            n_dict['cnt'][idx] += 1
        
        if c not in c_dict['c']:
            c_dict['c'].append(c)
            c_dict['cnt'].append(1)
        else:
            idx = c_dict['c'].index(c)
            c_dict['cnt'][idx] += 1

        if h not in h_dict['h']:
            h_dict['h'].append(h)
            h_dict['cnt'].append(1)
        else:
            idx = h_dict['h'].index(h)
            h_dict['cnt'][idx] += 1

        if w not in w_dict['w']:
            w_dict['w'].append(w)
            w_dict['cnt'].append(1)
        else:
            idx = w_dict['w'].index(w)
            w_dict['cnt'][idx] += 1

    n_pd = pd.DataFrame(n_dict)    
    c_pd = pd.DataFrame(c_dict)
    h_pd = pd.DataFrame(h_dict)
    w_pd = pd.DataFrame(w_dict)

    n_pd.sort_values(by='cnt', ascending=False, inplace=True)
    c_pd.sort_values(by='cnt', ascending=False, inplace=True)
    h_pd.sort_values(by='cnt', ascending=False, inplace=True)
    w_pd.sort_values(by='cnt', ascending=False, inplace=True)

    n_pd.to_csv('n.csv', index=False, header=True)
    c_pd.to_csv('c.csv', index=False, header=True)
    h_pd.to_csv('h.csv', index=False, header=True)
    w_pd.to_csv('w.csv', index=False, header=True)


def input_tensor_feature():
    column = 40
    in_sizes = get_input_data()
    all_size = len(in_sizes)

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
    for in_data in input_mem_data:
        mem_dict[label_data[int(np.floor((in_data-min_val)/step))]] += 1

    print('max input size ', np.max(input_mem_data))
    print('min input size ', np.min(input_mem_data))
    print('============mem===============\n', mem_dict)
    precent = np.array([value / all_size * 100 for value in mem_dict.values()])
    print('============mem percent============\n', precent)

def append_kernelinfo(perf_list, kernel_data_dict, metric_info, cnt, count_perf, rangecnt, idx_list):
    if len(kernel_data_dict['x1'][cnt]) != 4:
        return
    if cnt not in idx_list:
        return
    if kernel_data_dict['x1'][cnt][2] != kernel_data_dict['x1'][cnt][3]:
        return
    count_perf[rangecnt] += 1
    metric_info['perf'].append(perf_list[cnt])
    
    for j in range(len(arg)):
        metric_info[arg[j]].append(kernel_data_dict['x1'][cnt][j])

def write_csv(metric_info, range):
    datacsv = pd.DataFrame(metric_info)
    datacsv.sort_values(by=['N', 'C', 'H'], inplace=True)
    datacsv.to_csv('./data/perfrange'+range+'.csv', index=False, header=True)

def print_metricinfo(dfmetric, kerneljson, idx_list):
    header_list = arg + ['perf']
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
    perf_list = dfmetric['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].values
    print('lenofperf ', len(perf_list))

    for i in range(len(perf_list)):
        if perf_list[i] <= 20:
            append_kernelinfo(perf_list, kernel_data_dict, metricunder20_info, i, count_perf, 0, idx_list)
        elif perf_list[i] > 20 and perf_list[i] <= 30:
            append_kernelinfo(perf_list, kernel_data_dict, metric2030_info, i, count_perf, 1, idx_list)
        elif perf_list[i] > 30 and perf_list[i] <= 40:
            append_kernelinfo(perf_list, kernel_data_dict, metric3040_info, i, count_perf, 2, idx_list)
        elif perf_list[i] > 40 and perf_list[i] <= 50:
            append_kernelinfo(perf_list, kernel_data_dict, metric4050_info, i, count_perf, 3, idx_list)
        elif perf_list[i] > 50 and perf_list[i] <= 60:
            append_kernelinfo(perf_list, kernel_data_dict, metric5060_info, i, count_perf, 4, idx_list)
        elif perf_list[i] > 60 and perf_list[i] <= 70:
            append_kernelinfo(perf_list, kernel_data_dict, metric6070_info, i, count_perf, 5, idx_list)
        elif perf_list[i] > 70:
            append_kernelinfo(perf_list, kernel_data_dict, metricabove70_info, i, count_perf, 6, idx_list)

    
    write_csv(metricabove70_info, 'above70')
    write_csv(metric6070_info, '60-70')
    write_csv(metric5060_info, '50-60')
    write_csv(metric4050_info, '40-50')
    write_csv(metric3040_info, '30-40')
    write_csv(metric2030_info, '20-30')
    write_csv(metricunder20_info, 'under20')

    total_num = len(idx_list)
    stringrange = ['rangeabove70', 'range60-70', 'range50-60', 'range40-50', 'range30-40', 'range20-30', 'rangeunder20']
    print('total cases: ', total_num)
    for i in range(7):
        print(stringrange[7-i-1], ' ', count_perf[i]/total_num)

      
file = 'nsight_interpolate_5999.csv'
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
    # os.remove("tmp.csv")
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
        'dram__bytes.sum',
        'lts__t_bytes.sum',
        'l1tex__t_bytes.sum']

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
    dfmetric['Bandwidth'] = dfmetric['dram__bytes.sum']/dfmetric['Time']/1024/1024/1024
    dfmetric['bw_ratio'] = dfmetric['Bandwidth']/1555 * 100

    dfmetric.to_csv('pd.csv')
    dfs[tag]=dfmetric


    print('====================input feature=======================')
    input_tensor_feature()

    label_data = []
    utilization_range_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
    for key in utilization_range_dict.keys():
        label_data.append(key)
    
    apr_insize_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
    apr_insizes = get_input_data()
    max_util = 0
    total_cnt = 0
    idx_list = []
    for idx, util in enumerate(dfmetric['bw_ratio'].values):
        if name_list[idx] != 'nearest2d':
            continue
        idx_list.append(idx)
        total_cnt += 1
        if max_util < util:
            max_util = util
        if util < 20:
            key = 0
        elif util > 20 and util <=30:
            key = 1
        elif util > 30 and util <=40:
            key = 2
        elif util > 40 and util <=50:
            key = 3
        elif util > 50 and util <=60:
            key = 4
        elif util > 60 and util <=70:
            key = 5
        elif util >= 70:
            key = 6
        else:
            key = int(util // 10) - 1
        apr_insize_dict[label_data[key]].append(apr_insizes[idx])
    print(NUM_DEDU, len(dfmetric['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].values), len(apr_insizes))
    print(max_util)
    print('total cnt is ', total_cnt)
    for key in apr_insize_dict.keys():
        print('======================',key,'============================\n')
        if len(apr_insize_dict[key]) == 0:
            continue
        print(np.median(apr_insize_dict[key])*4.0/1024/1024, np.mean(apr_insize_dict[key])*4.0/1024/1024)
        print(len(apr_insize_dict[key])/total_cnt)

    print_metricinfo(dfmetric, './interpolate_all_top5740_dedu.json', idx_list)