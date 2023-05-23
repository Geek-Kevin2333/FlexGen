#!/bin/bash
# 这个shell脚本的作用是在一个分布式的计算环境中，运行一个Python脚本flexgen.dist_flex_opt。
# 这个Python脚本使用了mpirun工具来进行并行计算。
# 具体来说，脚本首先获取本机的IP地址，然后将其作为计算节点之一。
# 脚本中的N_GPUS和N_CORES_PER_GPU变量分别指定了每个节点的GPU数量和每个GPU上的核心数量。
# 脚本使用mpirun命令来启动并行计算，其中--oversubscribe选项表示允许在节点上超额使用资源。
# -H选项指定了所有的计算节点，--map-by选项指定了如何在计算节点上映射进程，--bind-to选项指定了如何将进程绑定到计算节点上的CPU核心。
# -x选项指定了环境变量OMP_NUM_THREADS的值，这个值用来控制每个进程使用的线程数。
# 最后，脚本传递了一系列参数给flexgen.dist_flex_opt脚本，包括模型、批大小、数据路径等。
# 同时，脚本使用pgrep、awk和xargs命令来查找并杀死正在运行的Python进程，以确保新的计算进程能够顺利启动。
# 在这个脚本中，与前一个脚本不同的是，传递给--model参数的模型是facebook/opt-6.7b，--gpu-batch-size参数的值是24，--percent参数的值是100 0 100 0 100 0。
MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=6

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=prefill.dist_flex_opt_prefill
PYTHON_MERGE_SCRIPT=benchmark.flexgen.merge_trace_file

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x
# --allow-run-as-root：允许使用 root 用户身份运行 mpirun 命令。
  # your_command：指定要在集群中运行的应用程序或脚本，替换为要运行的实际命令。
  #--mca btl_tcp_if_exclude lo,docker0：指定 mpirun 使用的网络接口，以便仅使用指定的接口进行通信。
  #--mca oob_tcp_if_exclude lo,docker0：指定仅使用指定的接口进行通信。
  #--map-by ppr:4:node:pe=6：指定进程到处理器的映射方式。
  #--oversubscribe：允许超配节点以允许更多进程运行。
  #-H 172.17.0.2：指定要在哪些节点上运行作业。
  #--bind-to core：将进程绑定到核心以提高性能。
  #-x OMP_NUM_THREADS=6：在进程环境中定义 OMP 环境变量，并设置线程数为 6。
  #/root/anaconda3/envs/flexgen/bin/python：指定要使用的 Python 解释器路径，替换为实际的 Python 解释器路径。
  #-m flexgen.dist_flex_opt：用于在指定的 Python 包中运行指定的模块。
  #--head-ip 172.17.0.2：指定头节点的 IP 地址。
  #--port 7777：指定头节点上使用的端口号。
  #--use-mpi：指定使用 MPI 进行通信。
  #--model facebook/opt-6.7b：指定使用的神经网络模型名称。
  #--gpu-batch-size 24：指定每批次使用的 GPU 数量。
  #--percent 100 0 100 0 100 0：指定灵活性图中每一层剪枝的比例。
  # parser.add_argument("--percent", nargs="+", type=int,
  #        default=[100, 0, 100, 0, 100, 0],
   #        help="Six numbers. They are "
   #         "the percentage of weight on GPU, "
   #         "the percentage of weight on CPU, "
   #         "the percentage of attention cache on GPU, "
   #         "the percentage of attention cache on CPU, "
   #         "the percentage of activations on GPU, "
   #         "the percentage of activations on CPU")
  #--comm-device cpu：指定使用的通信设备为 CPU。
  #--cut-gen-len 5：指定裁剪图生成器裁剪长度。
  #--path _DUMMY_：指定生成的裁剪模型的保存路径。
  #     Case("--model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 10 90 0 100 0 100 --gpu-batch-size 160 --num-gpu-batches 2 --cpu --debug fewer_batch", "", False),
mpirun --allow-run-as-root \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --num-gpu-batches 4\
    --use-mpi \
    --overlap True\
    --model facebook/opt-6.7b \
    --gpu-batch-size 24 \
    --percent 10 90 0 100 0 100 \
    --comm-device cpu \
    --cut-gen-len 24 \
    --path _DUMMY_

$PYTHON_EXEC merge_trace_file.py \
  --num-gpu-batches 4\
  --model facebook/opt-6.7b \
  --gpu-batch-size 24 \
  --percent 10 90 0 100 0 100 \
  --overlap True\
  --comm-device cpu \
  --cut-gen-len 24 \
  --path _DUMMY_