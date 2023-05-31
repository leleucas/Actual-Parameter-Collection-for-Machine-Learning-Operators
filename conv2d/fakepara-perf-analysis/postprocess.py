# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import numpy as np
import os
import numpy as np
import pandas as pd
import json
import csv
import sys

dfs={}
df_all = pd.DataFrame()
df_row = pd.DataFrame()
batchsize = sys.argv[1]
kernelsize = sys.argv[2]
algo = sys.argv[3]

count_aa = 0
shape_out = {"Batch_sizeN": [], "input_channel": [], "image_size_H": [], "image_size_W": [], 
"output_channel": [], "kernel_sizeR": [], "kernel_sizeS": [], "strideU": [], 
"strideV": [], "pad_h": [], "pad_w": [], "algo0":[], "algo1":[], "algo2":[], "algo3":[], "algo4":[], "algo5":[], 
"algo6":[], "algo7":[], "algoMin": [], algo+"AlgoMin":[], "kernel_nums": []}
arg = ["Batch_sizeN", "input_channel", "image_size_H", "image_size_W", 
"output_channel", "kernel_sizeR", "kernel_sizeS", "strideU", 
"strideV", "pad_h", "pad_w", "algo0", "algo1", "algo2", "algo3", "algo4", "algo5", 
"algo6", "algo7", "algoMin", algo+"AlgoMin", "kernel_nums"]

count_aa += 1
datadir='.'
files = ['./data/conv2d_nsight'+str(batchsize)+'_'+str(kernelsize)+'_processed.csv']
jsonfile = './data/conv2d_perf'+str(batchsize)+'_'+str(kernelsize)+'_processed_with_kernelnum.json'

CASR_NUM = 0

def append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric_info, cnt, count_perf, max_perf, min_perf, aihbm_list, rangecnt):
    count_perf[rangecnt] += 1
    max_perf[rangecnt] = max(max_perf[rangecnt], float(aihbm_list[cnt]))
    min_perf[rangecnt] = min(min_perf[rangecnt], float(aihbm_list[cnt]))
    metric_info['ai'].append(aihbm_list[cnt])
    metric_info['perf'].append(perf_list[cnt])
    metric_info['tc_perf'].append(tc_list[cnt])
    metric_info['time'].append(time_list[cnt])
    metric_info['Achieved Occupancy'].append(achieved_occupancy[cnt])
    metric_info['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].append(globalmem_throughput_list[cnt])
    metric_info['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'].append(l1_throughput_list[cnt])
    metric_info['lts__throughput.avg.pct_of_peak_sustained_elapsed'].append(l2_throughput_list[cnt])
    metric_info['sm__throughput.avg.pct_of_peak_sustained_elapsed'].append(sm_throughput_list[cnt])
    metric_info['l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed'].append(sharedmem_throughput_list[cnt])
    metric_info['l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum'].append(sharedmem_bankconflict_list[cnt])
    metric_info['Threads'].append(thread_list[cnt])
    metric_info['Registers Per Thread'].append(register_list[cnt])
    for j in range(len(arg)):
        metric_info[arg[j]].append(kernel_data_dict[arg[j]][cnt])

def write_csv(metric_info, range):
    datacsv = pd.DataFrame(metric_info)
    savepath = './data/' + str(batchsize) + '-' + str(kernelsize)
    if not os.path.exists(savepath): os.mkdir(savepath)
    datacsv.to_csv(savepath+'/perfrange'+range+'.csv', index=False, header=True)

def print_metricinfo(dfmetric, kerneljson):
    header_list = arg + ['time', 'ai', 'tc_perf', 'perf', 'Achieved Occupancy', \
	'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed', \
	'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'sm__throughput.avg.pct_of_peak_sustained_elapsed', \
	'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed', \
	'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 'Threads', 'Registers Per Thread']
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
    achieved_occupancy = dfmetric['Achieved Occupancy'].values
    globalmem_throughput_list = dfmetric['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].values
    l1_throughput_list = dfmetric['l1tex__throughput.avg.pct_of_peak_sustained_elapsed'].values
    l2_throughput_list = dfmetric['lts__throughput.avg.pct_of_peak_sustained_elapsed'].values
    sm_throughput_list = dfmetric['sm__throughput.avg.pct_of_peak_sustained_elapsed'].values
    sharedmem_throughput_list = dfmetric['l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed'].values
    sharedmem_bankconflict_list = dfmetric['l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum'].values
    thread_list = dfmetric['Threads'].values
    register_list = dfmetric['Registers Per Thread'].values
    max_perf_val = 0.0

    peak_perf = 19.5 * 1000
    for i in range(len(perf_list)):
        if perf_list[i] > max_perf_val:
            max_perf_val = perf_list[i]
        if abs(peak_perf - perf_list[i])/peak_perf <= 0.3:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metricabove70_info, i, count_perf, max_perf, min_perf, aihbm_list, 0)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.3 and abs(peak_perf - perf_list[i])/peak_perf <= 0.4:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric6070_info, i, count_perf, max_perf, min_perf, aihbm_list, 1)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.4 and abs(peak_perf - perf_list[i])/peak_perf <= 0.5:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric5060_info, i, count_perf, max_perf, min_perf, aihbm_list, 2)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.5 and abs(peak_perf - perf_list[i])/peak_perf <= 0.6:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric4050_info, i, count_perf, max_perf, min_perf, aihbm_list, 3)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.6 and abs(peak_perf - perf_list[i])/peak_perf <= 0.7:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric3040_info, i, count_perf, max_perf, min_perf, aihbm_list, 4)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.7 and abs(peak_perf - perf_list[i])/peak_perf <= 0.8:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metric2030_info, i, count_perf, max_perf, min_perf, aihbm_list, 5)
        elif abs(peak_perf - perf_list[i])/peak_perf > 0.8:
            append_kernelinfo(perf_list, tc_list, time_list, achieved_occupancy, globalmem_throughput_list, l1_throughput_list, l2_throughput_list, sm_throughput_list, sharedmem_throughput_list, sharedmem_bankconflict_list, thread_list, register_list, kernel_data_dict, metricunder20_info, i, count_perf, max_perf, min_perf, aihbm_list, 6)
    
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

with open(jsonfile, 'r') as f:
    shape_dict_row = json.load(f)
    CASR_NUM = len(shape_dict_row["Batch_sizeN"])
    for i in range(CASR_NUM):
        for j in range(len(arg)):
            shape_out[arg[j]].append(shape_dict_row[arg[j]][i])
    f.close()

for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    dfs[tag]=pd.DataFrame()
    with open(file,'r') as f:
        df = pd.read_csv(file)
        df_row = pd.concat([df_row, df], axis=0)

        dft=pd.DataFrame(df, columns=['ID', 'Kernel Name','Metric Name', 'Metric Value'])
        dft['Metric Value'] = pd.to_numeric(dft['Metric Value'].str.replace(r',',''))
        dfmetric=pd.pivot_table(dft, index=['ID'], columns=['Metric Name'], values=['Metric Value'])
        dfmetric.to_csv("a.csv")
        csv_reader = csv.reader(open("a.csv"))
        count = 0
        with open("b.csv", 'w') as f:
            csv_writer = csv.writer(f)
            for line in csv_reader:
                if count != 0 and count != 2:
                    csv_writer.writerow(line)
                count += 1
            f.close
        count -= 2
        dfmetric = pd.read_csv(open("b.csv"))

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
        'l1tex__t_bytes.sum',
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum',
        'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed',
        'Achieved Occupancy', 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
        'Registers Per Thread', 'Threads']

        df_dict = {i: []
        for i in df_list
        }
        cur_case = 0
        cur_line = 0
        while cur_case < CASR_NUM:
            kernel_num = shape_out['kernel_nums'][cur_case]
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

        dfmetric = pd.read_csv(open("deal.csv"))
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

        dfmetric['GFLOP/s'] = dfmetric['all FLOPs']/ dfmetric['Time'] /1000/1000/1000
        dfmetric['TC GFLOP/s'] = dfmetric['TC FLOPs']/ dfmetric['Time'] /1000/1000/1000
        para_csv = pd.DataFrame(shape_out)
        para_metric_csv = pd.concat([para_csv, dfmetric], axis=1)
        para_metric_csv.to_csv('./data/conv2d_para_metric'+str(batchsize)+'_'+str(kernelsize)+'.csv', index=False, header=True)
        df_all = pd.concat([df_all, dfmetric], axis=0)


dfm=df_all

print_metricinfo(dfm, jsonfile)