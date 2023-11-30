#!/bin/bash
#SBATCH -J TR-eform
#SBATCH -p kshdexclu04
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0  
#SBATCH --gres=dcu:4
#SBATCH -N 1  

base_log_dir=./log/$SLURM_JOB_ID

#SBATCH -o $base_log_dir/j.o  
#SBATCH -e $base_log_dir/j.e 

mkdir -p ./log
mkdir -p $base_log_dir
mkdir -p $base_log_dir/dmesg
mkdir -p $base_log_dir/debug
mkdir -p $base_log_dir/output

dmesg_log=$base_log_dir/dmesg
debug_log=$base_log_dir/debug
output_log=$base_log_dir/output
hostfile_path=$base_log_dir/hostfile

#srun hostname |sort |uniq -c |awk '{printf "%s slots=4\n",$2}' > ${hostfile_path}

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module add compiler/rocm/dtk-22.10.1
source activate TRCV

python train.py --dmesg_log $dmesg_log --debug_log $debug_log --output_log $output_log
