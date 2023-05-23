#!/bin/bash
#这个shell脚本的作用是在一个分布式的计算环境中，运行一个Python脚本flexgen.dist_flex_opt。
# 这个Python脚本使用了mpirun工具来进行并行计算。具体来说，脚本首先获取本机的IP地址，然后将其作为计算节点之一。
# 脚本中的N_GPUS和N_CORES_PER_GPU变量分别指定了每个节点的GPU数量和每个GPU上的核心数量。
# 脚本使用mpirun命令来启动并行计算，其中--oversubscribe选项表示允许在节点上超额使用资源。-H选项指定了所有的计算节点，
# --map-by选项指定了如何在计算节点上映射进程，--bind-to选项指定了如何将进程绑定到计算节点上的CPU核心。
# -x选项指定了环境变量OMP_NUM_THREADS的值，这个值用来控制每个进程使用的线程数。
# 最后，脚本传递了一系列参数给flexgen.dist_flex_opt脚本，包括模型、批大小、数据路径等。
# 同时，脚本使用pgrep、awk和xargs命令来查找并杀死正在运行的Python进程，以确保新的计算进程能够顺利启动。
MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=12

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=prefill.dist_flex_opt_prefill

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun --allow-run-as-root\
  --verbose\
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core:overload-allowed -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  --mca orte_base_help_aggregate 0\
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-175b \
    --gpu-batch-size 20 \
    --percent 0 100 0 100 0 100 \
    --comm-device cpu \
    --path _DUMMY_ \
    --cut-gen-len 5 \
    --pin-weight 0 \
    --cpu

$PYTHON_EXEC merge_trace_file.py \
  --num-gpu-batches 1\
  --model facebook/opt-175b \
  --overlap True\
  --gpu-batch-size 20 \
  --percent 0 100 0 100 0 100 \
  --comm-device cpu \
  --cut-gen-len 5 \
  --pin-weight 0 \
  --path _DUMMY_