import json
import pandas as pd
from enum import Enum
import os


digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class DataType(Enum):
    FLOAT = "float"
    FALSE = "flase"
    TRUE = "true"
    INT = "int"
    LIST = "list"

def string2list(s):
    ret = []
    s_l = len(s)
    if s_l == 2 and s[0:2] == "[]":
        return ret
        
    start = 1
    data_type = DataType("int")
    i = 0
    for i in range(s_l):
        if s[i] == ',':
            if data_type == DataType.INT:
                ret.append(int(s[start:i]))
            elif data_type == DataType.FLOAT:
                ret.append(float(s[start:i]))
            data_type = DataType("int")
            start = i + 2
        elif s[i] == '.' or s[i] == 'e':
            data_type = DataType("float")
    if data_type == DataType.INT:
        ret.append(int(s[start:s_l-1]))
    elif data_type == DataType.FLOAT:
        ret.append(float(s[start:s_l-1]))
    return ret

def parse_list(source_item, start_idx, info_length):
    assert source_item[start_idx] == '[', "parse list error"
    i = start_idx
    item = []
    while i < info_length:
        if source_item[i] == ']':
            item = string2list(source_item[start_idx:i+1])
            return i+3, item
        i += 1
    return i+1, item

def parse_num(source_item, start_idx, info_length):
    assert source_item[start_idx] in digits, "parse num error"
    
    i = start_idx + 1
    data_type = DataType("int")
    separators = ['/', '}', ',']

    while i < info_length:
        if source_item[i] == '.' or source_item[i] == 'e':
            data_type = DataType("float")
        elif source_item[i] in separators and data_type == DataType.INT:
            return i + 1, [int(source_item[start_idx:i])]
        elif source_item[i] in separators and data_type == DataType.FLOAT:
            return i + 1, [float(source_item[start_idx:i])]
        i += 1

    if data_type == DataType.INT:
        return i + 1, [int(source_item[start_idx:info_length])]
    else:
        return i + 1, [float(source_item[start_idx:info_length])]


# def process_data(filename):
#     data_csv = pd.read_csv(filename,header=None,names=['0','1','2','3','4'])
#     strdata_list = data_csv['2'].values
#     # print(len(strdata_list), len(strdata_list[1]))
#     # print(strdata_list[1][0])
#     info_length = len(strdata_list[1])
#     # print(info_length)
#     i = 0
#     data_list = []
#     while i < info_length:
#         if strdata_list[1][i] == 't' and strdata_list[1][i:i+12] == 'torch.Size([':
#             i = i + 11
#             i, val = parse_list(strdata_list[1], i, info_length)
#             assert(strdata_list[1][i+1:i+27] == "'torch.cuda.FloatTensor'}]")
#             i += 27
#             data_list.append(val)
#             # print(strdata_list[1][i+1:i+27])
#             # print(i, val)
#         i += 1

#     return data_list


def process_data(filename):
    data_csv = pd.read_csv(filename,header=None,names=['0','1','2','3','4'])
    strdata_list = data_csv['2'].values
    # print(len(strdata_list), len(strdata_list[1]))
    # print(strdata_list[1][0])
    info_length = len(strdata_list[1])
    # print(info_length)
    i = 0
    data_list = []
    while i < info_length:
        if strdata_list[1][i] == 't' and strdata_list[1][i:i+12] == 'torch.Size([':
            i = i + 11
            i, val = parse_list(strdata_list[1], i, info_length)
            assert(strdata_list[1][i] == ']')
            data_list.append(val)
            # print(strdata_list[1][i+1:i+27])
            # print(i, val)
        i += 1

    return data_list


def process_sam_data(filename):
    data_csv = pd.read_csv(filename,header=None,names=['0','1','2','3','4'])
    strdata_list = data_csv['3'].values
    # print(len(strdata_list), len(strdata_list[1]))
    # print(strdata_list[1][0])
    info_length = len(strdata_list[3])
    # print(info_length)
    i = 0

    flag = 0
    data_set = []
    feq_list = []
    q_list = []
    k_list = []
    v_list = []
    print('info_length ', info_length)
    
    while i < info_length:
        if strdata_list[3][i] == 't' and strdata_list[3][i:i+12] == 'torch.Size([':
            i = i + 11
            i, val = parse_list(strdata_list[3], i, info_length)
            # assert(strdata_list[3][i+1:i+27] == "'torch.cuda.FloatTensor'}]")
            # i += 27
            if flag == 0:
                q_list.append(val)
                flag = 1
            elif flag == 1:
                k_list.append(val)
                flag = 2
            elif flag == 2:
                v_list.append(val)
                assert(strdata_list[3][i] == ']' or \
                    strdata_list[3][i+1:i+27] == "'torch.cuda.FloatTensor'}]")
                ele = [q_list[-1], k_list[-1], v_list[-1]]
                # print('debug＝＝＝　', ele, i, '\n\n')
                if ele not in data_set:
                    data_set.append(ele)
                    feq_list.append(1)
                else:
                    idx = data_set.index(ele)
                    feq_list[idx] += 1
                flag = 0
            # print(strdata_list[1][i+1:i+27])
            # print(i, val)
        i += 1

    return data_set, feq_list


def gen_nsightres_processed(filename):
    df_list = [
        'sm__cycles_elapsed.avg',
        'sm__cycles_elapsed.avg.per_second',
        'sm__sass_thread_inst_executed_op_dfma_pred_on.sum', 
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
        'Achieved Occupancy','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum']

    data = pd.read_csv(filename)
    metric_list = data['Metric Name']
    for i in range(len(metric_list)):
        if metric_list[i] not in df_list:
            data.drop(index=i, inplace=True)
    data.to_csv(os.path.splitext(filename)[0]+'_processed.csv', index=False, header=True)


def print_para(filename):
    data = pd.read_csv(filename, usecols=['l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed', 
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum', 
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 
        'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
        'lts__throughput.avg.pct_of_peak_sustained_elapsed', 'Achieved Occupancy',
        'Time','sm__throughput.avg.pct_of_peak_sustained_elapsed'])
    
    data.to_csv(os.path.splitext(filename)[0]+'_metric_data.csv', header=True, index=False)

if __name__ == '__main__':
    # data_list = process_data('./data/sam_attention_vitl_raw.csv')
    # print(data_list, '\n\n')

    # data_list, feq_list = process_sam_data('./data/sam_attention_vith_raw.csv')
    # print('256, 8\n', data_list, '\n', feq_list)

    # gen_nsightres_processed('./data/5-9/bert_flashattn_nsight.csv')
    # gen_nsightres_processed('./data/5-9/bert_torch_nsight.csv')

    # gen_nsightres_processed('./data/5-9/gpt2_flashattn_nsight.csv')
    # gen_nsightres_processed('./data/5-9/gpt2_torch_nsight.csv')

    # gen_nsightres_processed('./data/5-9/sam_flashattn_nsight.csv')
    # gen_nsightres_processed('./data/5-9/sam_torch_nsight.csv')
    # gen_nsightres_processed('./data/5-9/sam7-4096_torch_nsight.csv')
    # gen_nsightres_processed('./data/5-9/sam4096-7_torch_nsight.csv')

    print_para('./data/5-9/res/sam_flashattn_nsight_processed_res.csv')
