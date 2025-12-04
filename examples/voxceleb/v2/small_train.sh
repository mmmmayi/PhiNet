###!/bin/bash

##./run.sh> exp/2pooling_constrainSamePho0.0007_diff0.00001/log 2>&1&


#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N test
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=20gb
#PBS -l walltime=72:00:00

cd $PBS_O_WORKDIR;

mkdir -p /scratch/e0643891
rsync -hav /hpctmp/e0643891/sitw /scratch/e0643891/

image="/app1/common/singularity-img/3.0.0/pytorch_2.0_cuda_12.0_cudnn8-devel_u22.04.sif"

singularity exec $image bash << EOF >  error.$PBS_JOBID 2> stdout.$PBS_JOBID
PYTHONPATH=$PYTHONPATH:/home/svu/e0643891/local/volta_pypkg/local/lib/python3.10/dist-packages/
export PYTHONPATH

./run.sh
EOF
