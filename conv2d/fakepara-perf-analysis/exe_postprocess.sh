kernelsize=1
for ((batchsize=16;batchsize<=128;batchsize*=2)); do
  python postprocess.py $batchsize $kernelsize gemm >metricinfo${batchsize}-${kernelsize}.dat
done
kernelsize=3
for ((batchsize=16;batchsize<=128;batchsize*=2)); do
  python postprocess.py $batchsize $kernelsize wino >metricinfo${batchsize}-${kernelsize}.dat
done
