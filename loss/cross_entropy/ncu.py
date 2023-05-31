# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
import collections
import json
import csv

datadir='.'
files=[x for x in os.listdir(datadir) if x.endswith('nsight.csv')]
files.sort()
files=[os.path.join(datadir,file) for file in files]
dfs={}


repeat_number = pd.read_csv(open("cross_entropy_repeat_number.csv"))
ALL_NUMBER = np.sum(repeat_number.values)
NUM_Metrics = 21
NUM_DEDU = len(repeat_number.values)

def get_dedu_input_data():
    with open("./cross_entropy_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    in_sizes = []
    label_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input"][i]:
            in_size *= dim
        in_sizes.append(in_size)
        label_sizes.append(arg_data["input"][i][1])
    return in_sizes, label_sizes

def input_tensor_feature():
    column = 28
    in_sizes,_ = get_dedu_input_data()
    all_size = len(in_sizes)

    input_mem_data = np.array([(i * 4.0 / 1024/ 1024) for i in in_sizes])
    step = int(np.ceil((np.max(input_mem_data) - np.min(input_mem_data)) / column))
    mem_dict = {}
    label_data = []

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


with open("cross_entropy_dedu.json", 'r') as f:
    shape_dict = json.load(f)

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
                        / (dfmetric['sm__cycles_elapsed.avg.per_second'] )
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
         'dram__bytes.sum',
         'lts__t_bytes.sum',
         'l1tex__t_bytes.sum']

        df_dict = {i: []
        for i in df_list
         }
        
        cur_line = 0
        kernel_keys = "softmax"
        # kernel_keys = "nll_loss"
        k_start = False
        for index, kernel in dfmetric.iterrows():
            if kernel_keys in kernel['Kernel Name'].lower():
                for j in range(len(df_list)):
                    curnum = 0.0
                    for i in range(cur_line, cur_line+1):
                        curnum += float(dfmetric[df_list[j]][i])
                    df_dict[df_list[j]].append(curnum)
            cur_line += 1

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


        # print(dfmetric['AI L2'].values)
        print(np.sum(dfmetric['AI L2'].values) / len(dfmetric['AI L2'].values))
        dfmetric.to_csv('pd.csv')
        dfs[tag]=dfmetric

        print('====================input feature=======================')
        input_tensor_feature()


        label_data = []
        utilization_range_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
        for key in utilization_range_dict.keys():
            label_data.append(key)
        
        apr_insize_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
        apr_insizes,_ = get_dedu_input_data()
        max_util = 0
        total_cnt = 0
        for idx, util in enumerate(dfmetric['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].values):
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
        print(NUM_DEDU, len(dfmetric['bw_ratio'].values), len(apr_insizes))
        print(max_util)
        print('total cnt is ', total_cnt)
        for key in apr_insize_dict.keys():
            print('======================',key,'============================\n')
            if len(apr_insize_dict[key]) == 0:
                continue
            print(np.median(apr_insize_dict[key])*4.0/1024/1024, np.mean(apr_insize_dict[key])*4.0/1024/1024)
            print(len(apr_insize_dict[key])/total_cnt)