#!/bin/bash
#这是一个运行分布式训练脚本的shell脚本，具体的作用是：
 #获取当前节点的IP地址，并且获取其他所有节点的IP地址，生成all_hosts变量，用于mpirun命令的指定；
 #执行远程命令获取所有节点的IP地址，并将所有的IP地址放到数组OTHERS_IPADDR中；
 #将当前节点和其他节点的IP地址合并到ALL_IPADDR数组中，并截取前N_NODES个IP地址，生成all_hosts；
 #定义变量N_GPUS、N_NODES和N_CORES_PER_GPU用于后续的mpirun命令；
 #定义变量PYTHON_EXEC和PYTHON_SCRIPT，PYTHON_EXEC指定Python解释器的路径，PYTHON_SCRIPT指定要执行的Python脚本；
 #通过pgrep和awk命令查找并杀死已经在运行的Python进程；
 #通过mpirun命令运行Python脚本，其中包括了多个参数，包括要运行的Python脚本、训练模型名称、批次大小、使用的通信设备类型、路径等。其中--use-mpi选项表明使用MPI并行训练。
N_GPUS=1
N_NODES=4
N_CORES_PER_GPU=16

MY_IPADDR=$(hostname -i)
all_public_ips=$(ray get-worker-ips ~/ray_bootstrap_config.yaml)
for s in $all_public_ips; do
    ssh -o StrictHostKeyChecking=no $s hostname -i > /tmp/$s.ip &
done
wait
for s in $all_public_ips; do
    OTHERS_IPADDR+=($(cat /tmp/$s.ip))
done
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=flexgen.dist_flex_opt

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-6.7b \
    --gpu-batch-size 24 \
    --percent 100 0 100 0 100 0 \
    --comm-device gpu \
    --cut-gen-len 5 \
    --path _DUMMY_
