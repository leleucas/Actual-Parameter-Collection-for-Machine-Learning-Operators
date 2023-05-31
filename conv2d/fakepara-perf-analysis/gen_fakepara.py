import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import sys
import os


def gen_para(batch_size, kernel_size):
    data_dict = {'N':[], 'C_in':[], 'H':[], 'W':[], 'C_out':[], 
                'kernel_R':[], 'kernel_S':[], 'strideU':[], 'strideV':[], 'pad_h':[], 'pad_w':[]}

    cin_cout_list = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 384, 512, 640, 768, 896, 1024, 2048, 4096]
    h_w_list = [1, 2, 4, 7, 8, 15, 16, 24, 32, 48, 64, 96, 97, 128, 256, 384, 512, 640, 768, 896, 1024, 2048, 3072]

    for u in cin_cout_list:
        for v in h_w_list:
            data_dict['N'].append(batch_size)
            data_dict['C_in'].append(u)
            data_dict['H'].append(v)
            data_dict['W'].append(v)
            data_dict['C_out'].append(u)
            data_dict['kernel_R'].append(kernel_size)
            data_dict['kernel_S'].append(kernel_size)
            data_dict['strideU'].append(1)
            data_dict['strideV'].append(1)
            data_dict['pad_h'].append(0)
            data_dict['pad_w'].append(0)
    
    return data_dict


def saveas_json(data_dict, batch_size, kernel_size):
    with open('./data/conv2d_fakepara'+str(batch_size)+'_'+str(kernel_size)+'.json', 'w') as f:
        json.dump(data_dict, f)


def saveas_json_byfilename(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f)


def saveas_csv(data_dict, batch_size, kernel_size):
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv('./data/conv2d_fakepara'+str(batch_size)+'_'+str(kernel_size)+'.csv', index=False, header=True)


def process_gemmperf_data(filename):
    data_pd = pd.read_csv(filename)
    algo_min = data_pd['algoMin'].values
    algo0 = data_pd['algo0'].values
    algo1 = data_pd['algo1'].values
    algo2 = data_pd['algo2'].values
    # algo1_num = pd.to_numeric(algo1)
    # algo2_num = pd.to_numeric(algo2)

    algo_gemm_min = []
    gemm_cnt = 0
    wino_cnt = 0
    total_cnt = 0
    filtered_gemm_cnt = 0
    filtered_wino_cnt = 0
    filtered_total_cnt = 0

    for idx, algo in enumerate(algo0):
        if algo_min[idx] == '6' or algo_min[idx] == '7':
            wino_cnt += 1
            total_cnt += 1
        elif algo_min[idx] == '0' or algo_min[idx] == '1' or algo_min[idx] == '2':
            gemm_cnt += 1
            total_cnt += 1
        else:
            total_cnt += 1

        if algo0[idx] == 'NOT SUPPORT!' and algo1[idx] == 'NOT SUPPORT!' and algo2[idx] == 'NOT SUPPORT!':
            data_pd.drop(index=idx, inplace=True)
            continue
        min_algo = 0
        if algo != 'NOT SUPPORT!':
            min_tm = float(algo)
        else:
            min_tm = sys.float_info.max
        
        if algo1[idx] != 'NOT SUPPORT!' and min_tm > pd.to_numeric(algo1[idx]):
            min_tm = float(pd.to_numeric(algo1[idx]))
            min_algo = 1
        
        if algo2[idx] != 'NOT SUPPORT!' and min_tm > pd.to_numeric(algo2[idx]):
            min_tm = float(pd.to_numeric(algo2[idx]))
            min_algo = 2

        if algo_min[idx] == '6' or algo_min[idx] == '7':
            filtered_wino_cnt += 1
            filtered_total_cnt += 1
        elif algo_min[idx] == '0' or algo_min[idx] == '1' or algo_min[idx] == '2':
            filtered_gemm_cnt += 1
            filtered_total_cnt += 1
        else:
            filtered_total_cnt += 1
        #     print('else branch', algo_min[idx])

        algo_gemm_min.append(min_algo)
    print('min gemm algo ratio: ', 'gemmcnt ', gemm_cnt, 'winocnt ', wino_cnt, 'totalcnt ', total_cnt, 
            'ratio ', gemm_cnt/total_cnt, 'processed_gemmcnt ', filtered_gemm_cnt, 'processed_winocnt ', filtered_wino_cnt, 
            'processed_totalcnt ', filtered_total_cnt, 'processed_ratio ', filtered_gemm_cnt/total_cnt,
            len(algo_gemm_min))
    data_pd['gemmAlgoMin'] = algo_gemm_min
    
    # data_pd.to_csv('newdatafile', index=False, header=True)
    data_dict = data_pd.to_dict('list')
    saveas_json_byfilename(data_dict, os.path.splitext(filename)[0]+'_processed.json')


def process_winoperf_data(filename):
    data_pd = pd.read_csv(filename)
    algo_min = data_pd['algoMin'].values
    algo6 = data_pd['algo6'].values
    algo7 = data_pd['algo7'].values
    # algo6_num = pd.to_numeric(algo6)
    # algo7_num = pd.to_numeric(algo7)

    algo_wino_min = []
    gemm_cnt = 0
    wino_cnt = 0
    total_cnt = 0
    filtered_gemm_cnt = 0
    filtered_wino_cnt = 0
    filtered_total_cnt = 0

    for idx, algo in enumerate(algo6):
        if algo_min[idx] == '6' or algo_min[idx] == '7':
            wino_cnt += 1
            total_cnt += 1
        elif algo_min[idx] == '0' or algo_min[idx] == '1' or algo_min[idx] == '2':
            gemm_cnt += 1
            total_cnt += 1
        else:
            total_cnt += 1

        if algo6[idx] == 'NOT SUPPORT!' and algo7[idx] == 'NOT SUPPORT!':
            data_pd.drop(index=idx, inplace=True)
            continue
        min_algo = 6
        if algo != 'NOT SUPPORT!':
            min_tm = float(algo)
        else:
            min_tm = sys.float_info.max
        
        if algo7[idx] != 'NOT SUPPORT!':
            if min_tm > pd.to_numeric(algo7[idx]):
                min_tm = float(pd.to_numeric(algo7[idx]))
                min_algo = 7

        if algo_min[idx] == '6' or algo_min[idx] == '7':
            filtered_wino_cnt += 1
            filtered_total_cnt += 1
        elif algo_min[idx] == '0' or algo_min[idx] == '1' or algo_min[idx] == '2':
            filtered_gemm_cnt += 1
            filtered_total_cnt += 1
        else:
            filtered_total_cnt += 1

        algo_wino_min.append(min_algo)
    print('min wino algo ratio: ', gemm_cnt, wino_cnt, total_cnt, wino_cnt/total_cnt,
            filtered_gemm_cnt, filtered_wino_cnt, filtered_total_cnt, filtered_wino_cnt/total_cnt)
    data_pd['winoAlgoMin'] = algo_wino_min
    
    data_dict = data_pd.to_dict('list')
    saveas_json_byfilename(data_dict, os.path.splitext(filename)[0]+'_processed.json')


def get_kernel_num(filename):
    kernel_nums = []
    with open(filename, 'r') as f:
        content = f.readline()
        while True:
            cnt = 0
            content = f.readline()
            if 'Disconnected' in content:
                break
            while '***' not in content:
                if '==PROF==' in content:
                    cnt += 1
                content = f.readline()
            kernel_nums.append(cnt)

        return kernel_nums

def gen_json_with_kernelnum(file1, kernel_nums):
    with open(file1, 'r') as f:
        data = json.load(f)
    data['kernel_nums'] = kernel_nums
    saveas_json_byfilename(data, os.path.splitext(file1)[0]+'_with_kernelnum.json')


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
        'Achieved Occupancy','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
        'Registers Per Thread', 'Threads']

    data = pd.read_csv(filename)
    metric_list = data['Metric Name']
    for i in range(len(metric_list)):
        if metric_list[i] not in df_list:
            data.drop(index=i, inplace=True)
    data.to_csv(os.path.splitext(filename)[0]+'_processed.csv', index=False, header=True)


if __name__ == '__main__':
    # batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    batch_size_list = [16, 32, 64, 128]
    
    for i in batch_size_list:
        # data_dict = gen_para(i, 1)
        # print(len(data_dict['N']))
        # saveas_json(data_dict, i, 1)
        # saveas_csv(data_dict, i, 1)
        # data_dict = gen_para(i, 3)
        # print(len(data_dict['N']))
        # saveas_json(data_dict, i, 3)
        # saveas_csv(data_dict, i, 3)

        # process_gemmperf_data('./data/conv2d_perf'+str(i)+'_1.csv')
        # process_winoperf_data('./data/conv2d_perf'+str(i)+'_3.csv')

        # kernel_nums = get_kernel_num('./data/kernel_numbers'+str(i)+'_1.dat')
        # gen_json_with_kernelnum('./data/conv2d_perf'+str(i)+'_1_processed.json', kernel_nums)

        # kernel_nums = get_kernel_num('./data/kernel_numbers'+str(i)+'_3.dat')
        # gen_json_with_kernelnum('./data/conv2d_perf'+str(i)+'_3_processed.json', kernel_nums)

        gen_nsightres_processed('./data/conv2d_nsight'+str(i)+'_1.csv')
        gen_nsightres_processed('./data/conv2d_nsight'+str(i)+'_3.csv')
