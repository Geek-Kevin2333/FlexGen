#!/bin/bash
#这是一个Shell脚本，用于在分布式集群上运行Python脚本flexgen.dist_flex_opt。
 #脚本首先获取当前机器的IP地址，并将其存储在变量MY_IPADDR中。然后，all_hosts变量被设置为当前机器的IP地址。
 #接下来，N_GPUS和N_CORES_PER_GPU变量被设置为4和12，分别表示每个节点上的GPU数量和每个GPU上的CPU核心数。
 #然后，PYTHON_EXEC和PYTHON_SCRIPT变量被设置为Conda环境中的Python解释器和flexgen.dist_flex_opt Python脚本的路径。
 #pgrep命令用于查找正在运行的Python进程，并使用awk和xargs命令将与flexgen.dist_flex_opt.py文件不匹配的进程杀死。
 #接下来，mpirun命令用于启动MPI进程，并在所有节点上执行flexgen.dist_flex_opt.py脚本。此命令将在每个节点上启动4个进程，每个进程使用12个CPU核心和1个GPU，并使用MPI通信协议进行通信。该命令使用MPI进程来实现并行计算。
 #脚本的最后几行指定了一些参数，如--model，--gpu-batch-size，--percent，--comm-device，--path和--cut-gen-len，这些参数将被传递给flexgen.dist_flex_opt Python脚本。这些参数用于指定模型、GPU批量大小、数据分区等等。其中，--comm-device设置为cpu表示使用CPU作为通信设备，--path指定了模型文件的路径，--cut-gen-len指定了切割长度。
MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=12

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=profile.dist_flex_opt_profile

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun --allow-run-as-root\
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core:overload-allowed -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  --mca orte_base_help_aggregate 0\
  --verbose\
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-30b \
    --gpu-batch-size 12 \
    --num-gpu-batches 3\
    --percent 20 80 0 100 0 100 \
    --comm-device cpu \
    --compress-weight \
    --compress-cache \
    --path _DUMMY_ \
    --cut-gen-len 2 \
    --cpu

$PYTHON_EXEC merge_trace_file.py \
  --num-gpu-batches 3\
  --model facebook/opt-30b \
  --overlap True\
  --compress-weight True\
  --compress-cache True\
  --gpu-batch-size 12 \
  --percent 20 80 0 100 0 100 \
  --comm-device cpu \
  --cut-gen-len 2 \
  --path _DUMMY_