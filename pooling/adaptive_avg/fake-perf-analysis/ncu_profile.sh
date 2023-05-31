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

#l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed    for wavefronts 
#l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum

#--section "MemoryWorkloadAnalysis" --section "ComputeWorkloadAnalysis"
# ncu --section "LaunchStats" --section "Occupancy" --metrics $metrics,group:memory__shared_table --csv --target-processes all ./conv2d_nsight_profile >conv2d_result.csv
# ncu --metrics group:memory__shared_table --csv --target-processes all ./conv2d_nsight_profile >conv2d_result_shared.csv
# ncu --metrics group:memory__first_level_cache_table --csv --target-processes all ./conv2d_nsight_profile >conv2d_result_l1.csv
# ncu --metrics group:memory__l2_cache_table --csv --target-processes all ./conv2d_nsight_profile >conv2d_result_l2.csv
# ncu --section "SpeedOfLight_RooflineChart" --csv --target-processes all ./conv2d_nsight_profile >roofline_result.csv
# batchsize=16
# /usr/local/cuda-11.2/bin/ncu --section "LaunchStats" --section "Occupancy" --metrics $metrics,group:memory__shared_table --csv --target-processes all python pooling.py > pooling_nsight${batchsize}.csv
for ((batchsize=64;batchsize<=64;batchsize*=2)); do
   /usr/local/cuda/bin/ncu --section "LaunchStats" --section "Occupancy" --metrics $metrics,group:memory__shared_table --csv --target-processes all python pooling.py $batchsize > ./data/pooling_nsight${batchsize}.csv
   # /usr/local/cuda/bin/ncu -o pooling --target-processes all -f python -u pooling.py $batchsize > ./data/kernel_numbers${batchsize}.dat
   # ncu -o conv2d --target-processes all -f ./conv2d_nsight_profile $batchsize 1 > ./data/kernel_numbers${batchsize}_1.txt
done
