import os
import numpy as np
import pandas as pd
# from torch import cudnn_convolution
# from roofline import roofline
import json
import csv

dfs={}
df_all = pd.DataFrame()
df_row = pd.DataFrame()
# shape_out = {"input_size": [], "normalized_shape": [], "weight_size": [], "bias_size": [], "eps": []}
# arg = ["input_size", "normalized_shape", "weight_size", "bias_size", "eps"]

# shape_out = {"N": [], "C_in": [], "H": [], "W": [], "C_out": [], "kernel_R": [], "kernel_S": [], "strideU": [], "strideV": [], "pad_h": [], "pad_w": [], "algoMin": [], "kernel_nums": [], "count": []}
# arg = ["N", "C_in", "H", "W", "C_out", "kernel_R", "kernel_S", "strideU", "strideV", "pad_h", "pad_w", "algoMin", "kernel_nums", "count"]
# count_aa = 0

# nsight_row = open('nsight_row.csv')
# kernel_nums_all = open('kernel_nums.csv')

# count_aa += 1
datadir='.'
files = ['./data/5-9/sam_flashattn_nsight_processed.csv']
# jsonfile = './top4349_count.json'
# files=["/home/liuhangda/bupt/opinsight/DL-Op-Analysis-cudnn_spec/Norm/layer_norm/layer_norm_nsight.csv"]
# kernel_nums = {"0": 1, "1": 2, "2": 2, "3": 0, "4": 5, "5": 4, "6": 2, "7": 4}

# CASR_NUM = 0

# repeat_number = pd.read_csv(open("/home/liuhangda/bupt/opinsight/DL-Op-Analysis-cudnn_spec/Norm/layer_norm/layer_norm_repeat_number.csv"))

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
        # print(dfmetric, '\n-------------------------\n')
        dfmetric = pd.read_csv(open("b.csv"))
        # dfmetric.drop(index=0, inplace=True)
        # dfmetric.drop(index=1, inplace=True)

        # print(dfmetric)
        # exit()

        # print(dft['Kernel Name'].values)
        kernelname_list = []
        for i in range(9):
            kernelname_list.append(dft['Kernel Name'].values[i*23])
            print(kernelname_list[-1],'\n\n')
        # print(kernelname_list)
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
        'Achieved Occupancy', 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum']

        df_dict = {i: []
        for i in df_list
        }
        cur_case = 0
        cur_line = 0
        CASR_NUM = len(dfmetric['Time'].values)
        while cur_case < CASR_NUM:
            kernel_num = 1
            for j in range(len(df_list)):
                curnum = 0.0
                for i in range(cur_line, cur_line+kernel_num):
                    curnum += float(dfmetric[df_list[j]][i])
                df_dict[df_list[j]].append(curnum)
            cur_line += kernel_num
            cur_case += 1
            # dfmetric['kernel_nums'][cur_case] = kernel_num

        header = df_dict.keys()
        rows=pd.DataFrame(df_dict).to_dict('records')
        # dfmetric.to_csv('metric.csv', index=False)
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
        dfmetric['kernelname'] = kernelname_list
        # dfmetric['kernel_nums'] = dfmetric['TC FLOPs']/ dfmetric['Time'] /1000/1000/1000
        # for cur_case in range(50):
        #     dfmetric['kernel_nums'][cur_case] = 2

        dfs[tag]=dfmetric
        df_all = pd.concat([df_all, dfmetric], axis=0)
        df_all.to_csv(os.path.splitext(file)[0]+'_res.csv', index=False, header=True)



# tags=dfs.keys()
# flags=['HBM'] #'HBM','L2','L1' or 'all'
# dfm=df_all
# df_all.to_csv('./res.csv', index=False)
# df_row.to_csv('./pd_a100_tc_row.csv', index=False)

# dump new json
# shape_out_json = json.dumps(shape_out)
# with open('a100_269.json', 'w') as f:
#     f.write(shape_out_json)

# with open('./a100_269.json', 'r') as f:
#     shape_dict_row = json.load(f)
#     print(CASR_NUM)
#     print(count_aa*50)

# print_metricinfo(dfm, jsonfile)

# LABELS = [str(i) for i in range(len(dfm.index.tolist()))]
# AIL1   = dfm['AI L1'].tolist()
# AIL2   = dfm['AI L2'].tolist()
# AIHBM  = dfm['AI HBM'].tolist()
# FLOPS  = dfm['GFLOP/s'].tolist()


# # roofline("", FLOPS, AIHBM, AIL2, AIL1, LABELS, flags[0], repeat_number)
# roofline("", FLOPS, AIHBM, AIL2, AIL1, LABELS, flags[0])