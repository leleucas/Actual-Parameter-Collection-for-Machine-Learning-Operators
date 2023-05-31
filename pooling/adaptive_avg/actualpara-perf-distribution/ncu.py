# Modified from Roofline-on-NVIDIA-GPUs with the url https://gitlab.com/NERSC/roofline-on-nvidia-gpus

import os
import numpy as np
import pandas as pd
import collections
import json
import csv
import math

datadir='.'
end = 'adaptive_avg_pool2d_nsight.csv'
files=[x for x in os.listdir(datadir) if x.endswith(end)]
files.sort()
files=[os.path.join(datadir,file) for file in files]
dfs={}


repeat_number = pd.read_csv(open("adaptive_avg_pool2d_repeat_number.csv"))
ALL_NUMBER = np.sum(repeat_number.values)
NUM_Metrics = 21
NUM_DEDU = len(repeat_number.values)

def get_input_data():
    with open("./adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["x1"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_data():
    with open("./adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input_size"][i]:
            in_size *= dim
        in_sizes.append(in_size)
    return in_sizes

def get_dedu_input_info():
    with open("./adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_infos = []
    for i in range(arg_data_length):
        in_info = str(arg_data["input_size"][i][0])
        for dim in range(len(arg_data["input_size"][i])-1):
            in_info += "x" + str(arg_data["input_size"][i][dim+1])
        in_infos.append(in_info)
    return in_infos

def get_input_output_data():
    with open("adaptive_avg_pool2d_dedu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    in_sizes = []
    out_sizes = []
    ratio_list = []
    for i in range(arg_data_length):
        in_size = 1
        for dim in arg_data["input_size"][i]:
            in_size *= dim
        assert len(arg_data["input_size"][i]) == 4
        assert len(arg_data["output_size"][i]) == 2
        out_size = arg_data["input_size"][i][0] * arg_data["input_size"][i][1] * arg_data["output_size"][i][0] * arg_data["output_size"][i][1]
        in_sizes.append(in_size)
        out_sizes.append(out_size)
        ratio_list.append(in_size/out_size)
    return in_sizes, out_sizes, ratio_list

def print_pretty(d):
    print(json.dumps(d, indent=4, ensure_ascii=False))

with open("adaptive_avg_pool2d_dedu.json", 'r') as f:
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
        # kernel_keys = "adaptive_avg_pool2d_kernel_cuda"
        kernel_keys = "at::native::"
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


        dfmetric.to_csv('pd.csv')
        dfs[tag]=dfmetric

        in_infos = get_dedu_input_info()
        sizes  = get_dedu_input_data()
        sizes = [i*4.0/1024/1024 for i in sizes]

        min_insize = min(sizes)
        max_insize = max(sizes)
        num_of_ticks = 20
        stride_insize = (max_insize - min_insize) // num_of_ticks

        in_size_labels = [min_insize+stride_insize*i for i in range(num_of_ticks)]

        print('min size ', min_insize)
        
        if in_size_labels[-1] < max_insize:
            in_size_labels.append(in_size_labels[-1] + stride_insize)
        len_of_insize = 10
        
        new_max_insize = in_size_labels[len_of_insize - 1]
        num_of_ticks = 20
        new_stride_insize = (new_max_insize - min_insize) // num_of_ticks
        new_insize_labels = [min_insize+new_stride_insize*i for i in range(num_of_ticks)]
        if new_insize_labels[-1] < new_max_insize:
            new_insize_labels.append(new_insize_labels[-1] + new_stride_insize)
        
        len_of_insize = len(new_insize_labels)


        print('--------------------------------------------------------------------')
        print(new_insize_labels)


        label_data = []
        utilization_range_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}

        for key in utilization_range_dict.keys():
            utilization_range_dict[key] = np.zeros(len_of_insize)
            label_data.append(key)

        apr_insize_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
        apr_outsize_dict = {'0-20':[], '20-30':[], '30-40':[], '40-50':[], '50-60':[], '60-70':[], 'above70':[]}
        apr_insizes, apr_outsizes, ratio = get_input_output_data()

        for idx, util in enumerate(dfmetric['bw_ratio'].values):
            if util < 20:
                key = 0
            elif util >= 70:
                key = 6
            else:
                key = int(util // 10) - 1
            apr_insize_dict[label_data[key]].append(apr_insizes[idx])
            apr_outsize_dict[label_data[key]].append(apr_outsizes[idx])
        print(NUM_DEDU, len(dfmetric['gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'].values))
        for key in apr_insize_dict.keys():
            print('======================',key,'============================\n')
            print(np.median(apr_insize_dict[key])*4.0/1024/1024, np.mean(apr_insize_dict[key])*4.0/1024/1024)
            print(np.median(apr_outsize_dict[key])*4.0/1024/1024, np.mean(apr_outsize_dict[key])*4.0/1024/1024)
            print(len(apr_insize_dict[key])/NUM_DEDU)


        for idx, util in enumerate(dfmetric['bw_ratio'].values):
            if util < 20:
                key = 0
            elif util >= 70:
                key = 6
            else:
                key = int(util // 10) - 1
            insize = sizes[idx]
            if insize >= new_insize_labels[-1]:
                continue
            utilization_range_dict[label_data[key]][math.floor((insize - min_insize)/new_stride_insize)] += 1
        
        for key in utilization_range_dict:
            for i in range(len(utilization_range_dict[key])):
                utilization_range_dict[key][i] = utilization_range_dict[key][i]/NUM_DEDU * 100


        for item in utilization_range_dict.items():
            print(item)


        import matplotlib.pyplot as plt
        from pylab import xticks, yticks, np

        fig, ax = plt.subplots()
        plt.xlim(0, max(new_insize_labels))
        plt.ylim(0, 60)
        xticks_labels = ['{}'.format(5 * i) for i in range(0,9)]
        xticks(np.linspace(0,40*1024,9,endpoint=True), xticks_labels, fontsize=10)
        h = ax.stackplot(new_insize_labels, utilization_range_dict.values(),
                    labels=utilization_range_dict.keys(), alpha=0.8)

        legend_value = ["Under20%          1.5 / 7.9               0.01 / 0.3", 
                        "20%-30%           19.1 / 66.1           0.1 / 0.4", 
                        "30%-40%           13.5 / 39.2           0.008 / 0.2", 
                        "40%-50%           12.5 / 64.1           0.01 / 0.2", 
                        "50%-60%           16 / 17                 0.001 / 0.03", 
                        "60%-70%           28.7 / 28.9           0.002 / 0.06", 
                        "Above70%         68 / 92                 0.01 / 0.01"]
        leg = ax.legend(h, legend_value, loc='upper right', ncol=1, shadow=False, fancybox=False, prop={'size':9})
        leg.set_title("        MEMBand           input tensor        output tensor", prop={'size':9})
        leg._legend_box.align = "left"
        ax.set_xlabel('input data size (MB)')

        ax.set_ylabel('Actual Parameter Ratio(%)')
        plt.savefig('pooling.perf.png', format='png', bbox_inches='tight')
        plt.savefig('pooling.perf.pdf', format='pdf', bbox_inches='tight')