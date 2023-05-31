make
for ((batchsize=16;batchsize<=128;batchsize*=2)); do
   # echo $batchsize
   ./gen_fakepara $batchsize 1 >> res.txt
done

for ((batchsize=16;batchsize<=128;batchsize*=2)); do
   # echo $batchsize
   ./gen_fakepara $batchsize 3 >> res.txt
done
