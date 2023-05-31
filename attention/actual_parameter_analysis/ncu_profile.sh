# make
metrics="sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum,\
dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

/usr/local/cuda-11.4/bin/ncu --section "LaunchStats" --section "Occupancy" --metrics $metrics,group:memory__shared_table --csv --target-processes all python multihead_atten.py > ./data/5-9/bert_torch16_nsight.csv