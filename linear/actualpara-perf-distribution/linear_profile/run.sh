#/bin/bash 
# Time
export PYTHONPATH=/tmp/AApity/DL-Op-Analysis-main/common:$PYTHONPATH
# export NVIDIA_TF32_OVERRIDE=0

metrics="sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"

nohup /usr/local/cuda-11.1/bin/ncu --metrics $metrics --csv --target-processes all python3 ./profile.py > nsight1381.csv &
# nohup /usr/local/cuda-11.1/bin/ncu --metrics $metrics --csv --target-processes all python3 ./profile_linear_v.py > nsightV.csv &
# nohup /usr/local/cuda-11.1/bin/ncu -o linear --set full --metrics $metrics -f python3 -u ./profile.py > hmc.txt &
