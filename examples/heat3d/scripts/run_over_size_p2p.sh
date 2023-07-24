#/bin/bash
BENCHMARK=$1
HOST=$2
DEFAULT_SIZE=1000

#exports
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export OMP_NUM_THREADS=32

ITERS=30

DS=$DATA_SIZE
#print header
HASH=`date|md5sum|head -c 5`
FILENAME="${BENCHMARK}_${HASH}_p2p.res"
echo $FILENAME
echo "name,type,N,size,iters,time,gups,bw" | tee $FILENAME 
VARS0="-x CUDA_VISIBLE_DEVICES=0,1 -x NVSHMEM_SYMMETRIC_SIZE=10737418240"
VARS1="-x UCX_WARN_UNUSED_ENV_VARS=n -x HCOLL_MAIN_IB=enP48p1s0f0"
VARS2="-x LD_LIBRARY_PATH=/projects/ppc64le-pwr9-rhel8/tpls/cuda/11.8.0/gcc/11.3.0/base/ztdfrze/lib64/:$LD_LIBRARY_PATH"

#run test over size
SIZE=$DEFAULT_SIZE
for S in $(seq 1 21); do 
   for reps in $(seq 1 3); do
      mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 1 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done

#run test over size
let SIZE=$DEFAULT_SIZE
for S in $(seq 1 21); do 
   for reps in $(seq 1 3); do
      mpirun -np 2 $VARS0 $VARS1 $VARS2 -host $HOST  ./$BENCHMARK -N $SIZE -I $ITERS -M 0 | tee -a $FILENAME
   done
   let SIZE=$SIZE*2
done
