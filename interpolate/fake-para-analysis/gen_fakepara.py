import json
import pandas as pd
import os

def process_data(filename):
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
    # batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # for ele in batch_size:
    #     gen_fakapara(ele)

    process_data('./nearest_nsightdat.csv')