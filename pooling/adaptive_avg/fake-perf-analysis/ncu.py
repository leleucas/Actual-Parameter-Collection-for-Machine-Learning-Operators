# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
import collections
import json
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

datadir='./data/'

end = 'pooling_nsight64_processed.csv'
end_kernel_num = 'kernel_numbers64.dat'

files=[x for x in os.listdir(datadir) if x.endswith(end)]
files.sort()
files=[os.path.join(datadir,file) for file in files]
dfs={}

NUM_Metrics = 25

def get_input_output_data():
    with open("adaptive_avg_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    kernel_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        in_HW_size = 1
        idx = 0
        for dim in arg_data["input_size"][i]:
            in_size *= dim
            if idx == 2 or idx == 3: # HW
                in_HW_size *= dim
            idx += 1
        in_sizes.append(in_size)
        out_size = 1
        for dim in arg_data["output_size"][i]:
            out_size *= dim
        out_sizes.append(out_size * in_size/ in_HW_size)
        kernel_sizes.append(in_HW_size/out_size)
    return in_sizes, out_sizes, kernel_sizes

def get_dedu_input_data():
    with open("adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    kernel_sizes = []
    for i in range(arg_data_length):
        if arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["input_size"][i][2] * arg_data["input_size"][i][3]:
            in_size = 1
            in_HW_size = 1
            idx = 0
            for dim in arg_data["input_size"][i]:
                in_size *= dim
                if idx == 2 or idx == 3: # HW
                    in_HW_size *= dim
                idx += 1
            out_size = 1
            for dim in arg_data["output_size"][i]:
                out_size *= dim
            in_sizes.append(in_size)
            out_sizes.append(out_size)
            kernel_sizes.append(in_size/in_HW_size)
    return in_sizes, out_sizes, kernel_sizes

def get_dedu_input_info():
    with open("adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_infos = []
    out_infos = []
    kernel_infos = []
    for i in range(arg_data_length):
        if arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["input_size"][i][2] * arg_data["input_size"][i][3] :
            in_info = str(arg_data["input_size"][i][0])
            for dim in range(len(arg_data["input_size"][i])-1):
                in_info += "x" + str(arg_data["input_size"][i][dim+1])
            in_infos.append(in_info)
            out_info =  str(arg_data["output_size"][i][0]) + "x" + str(arg_data["output_size"][i][1])
            out_infos.append(out_info)
            
            kernel_info = in_info + " " + str(arg_data["output_size"][i][0]) + " " + str(arg_data["output_size"][i][1])
            NC_size =  arg_data["input_size"][i][0] * arg_data["input_size"][i][1]
            kernel_infos.append(kernel_info)
    return in_infos,out_infos,kernel_infos


with open("./data/pooling_fakepara64.json", 'r') as f:
    shape_dict = json.load(f)

NUM_DEDU = len(shape_dict['input_size'])

kernel_num = []
with open(datadir+end_kernel_num, 'r') as f:
    line_kernelnum = f.readline()
    kernel_nums = []
    while True:
        count = 0
        line_kernelnum = f.readline()
        if 'Disconnected' in line_kernelnum:
            break
        while '***' not in line_kernelnum:
            if 'PROF' in line_kernelnum:
                count += 1
            line_kernelnum = f.readline()
        kernel_nums.append(count)
    kernel_nums[0] -= 1
    f.close()

print('case num ', NUM_DEDU)

for file in files:
    tag, ext = os.path.splitext(os.path.basename(file))
    dfs[tag]=pd.DataFrame()
    with open(file,'r') as f:
        df = pd.read_csv(file)
        df_group=pd.DataFrame()
        dft=pd.DataFrame(df, columns=['ID', 'Kernel Name','Metric Name', 'Metric Value'])
        
        dft['Metric Value'] = pd.to_numeric(dft['Metric Value'].str.replace(r',',''))
        dfmetric=pd.pivot_table(dft, index=['ID'], columns=['Metric Name'], values=['Metric Value'])
        dfmetric.to_csv("tmp.csv")
        kernel_name = dft['Kernel Name']
        kernel_name_dedu = []
        for k in range(0,len(kernel_name),NUM_Metrics):
            kernel_name_dedu.append(kernel_name[k])
        df_kernel_name_dedu=pd.DataFrame(kernel_name_dedu,columns=['Kernel Name'])

    
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
                        / (dfmetric['sm__cycles_elapsed.avg.per_second'])
        dfmetric['Kernel Name'] = df_kernel_name_dedu['Kernel Name']


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

        df_dict = {i: []
        for i in df_list
         }
        
        cur_line = 0
        kernel_keys = "at::native::"
        k_start = False

        cur_case = 0
        cur_line = 0
        while cur_case < NUM_DEDU:
            kernel_num = kernel_nums[cur_case]
            for j in range(len(df_list)):
                curnum = 0.0
                for i in range(cur_line, cur_line+kernel_num):
                    curnum += float(dfmetric[df_list[j]][i])
                df_dict[df_list[j]].append(curnum)
            cur_line += kernel_num
            cur_case += 1

        assert len(df_dict['Time']) == NUM_DEDU
        
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

        para_csv = pd.DataFrame(shape_dict)
        para_metric_csv = pd.concat([para_csv, dfmetric], axis=1)

        para_metric_csv.to_csv('./data/pooling_para_metric64.csv', index=False, header=True)