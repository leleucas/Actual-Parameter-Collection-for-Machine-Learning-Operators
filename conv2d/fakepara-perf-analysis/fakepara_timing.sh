#for ((batchsize=1;batchsize<=64;batchsize*=2)); do
#   # echo $batchsize
#   ./gen_fakepara $batchsize 1
#done

for ((batchsize=1;batchsize<=64;batchsize*=2)); do
   # echo $batchsize
   ./gen_fakepara $batchsize 3
done
