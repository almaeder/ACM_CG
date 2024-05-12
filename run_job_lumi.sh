#!/bin/bash -l
#SBATCH --job-name=hybrid-small # Job name
#SBATCH --partition=standard-g  # or ju-standard-g, partition name
#SBATCH --nodes=1               # Total number of nodes  - 1s
#SBATCH --ntasks-per-node=1    # 8 MPI ranks per node, 8 total (1x8) - 8
#SBATCH --gpus-per-node=1      # Allocate one gpu per MPI rank - 8
#SBATCH --cpus-per-task=7       # 7 cpus per task
#SBATCH --time=00:20:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000929

# The carefully assembled compile and runtime environment (DO NOT CHANGE ORDER)...:
# module restore kmc_env                                                                      
# module load craype-x86-trento                                                               
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"     
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)                                   

export MPICH_GPU_SUPPORT_ENABLED=1
export GMX_FORCE_GPU_AWARE_MPI=1
export GMX_ENABLE_DIRECT_GPU_COMM=1

export MPICH_OFI_NIC_POLICY=GPU

export HCC_AMDGPU_TARGET=gfx90a

export OMP_NUM_THREADS=1


if [ $SLURM_NTASKS_PER_NODE -eq 8 ]; then
    # 8 GPUs per node
    CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
    CPU_BIND="${CPU_BIND},fe0000,fe000000"
    CPU_BIND="${CPU_BIND},fe,fe00"
    CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 7 ]; then
    # 7 GPUs per node
    CPU_BIND="mask_cpu:0x00000000000000FE,0x00000000000000FE"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 6 ]; then
    # 6 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE000000,0xFE000000"
elif [ $SLURM_NTASKS_PER_NODE -eq 5 ]; then
    # 5 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE0000,0xFE0000"
elif [ $SLURM_NTASKS_PER_NODE -eq 4 ]; then
    # 4 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE00,0xFE00"
elif [ $SLURM_NTASKS_PER_NODE -eq 3 ]; then
    # 3 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
    CPU_BIND="${CPU_BIND},0xFE,0xFE"
elif [ $SLURM_NTASKS_PER_NODE -eq 2 ]; then
    # 2 GPUs per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
    CPU_BIND="${CPU_BIND},0x0000FE,0x0000FE"
elif [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
    # 1 GPU per node
    CPU_BIND="mask_cpu:0x0000000000000000,0x0000000000000000"
    CPU_BIND="${CPU_BIND},0x000000FE,0x000000FE"
else
    echo "Unsupported number of GPUs per node!"
    exit 1
fi

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF
chmod +x ./select_gpu

# srun ./select_gpu build/test
srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/test
# srun --cpu-bind=${CPU_BIND} ./wrapper.sh --hip-trace ./select_gpu ./build/test
