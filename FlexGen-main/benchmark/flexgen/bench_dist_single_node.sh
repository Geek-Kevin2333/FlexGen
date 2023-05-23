#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=4

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=flexgen.dist_flex_opt
ben
pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x
  # parser.add_argument("--percent", nargs="+", type=int,
  #        default=[100, 0, 100, 0, 100, 0],
   #        help="Six numbers. They are "
   #         "the percentage of weight on GPU, "
   #         "the percentage of weight on CPU, "
   #         "the percentage of attention cache on GPU, "
   #         "the percentage of attention cache on CPU, "
   #         "the percentage of activations on GPU, "
   #         "the percentage of activations on CPU")
mpirun --allow-run-as-root \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-30b \
    --gpu-batch-size 16 \
    --percent 10 90 10 90 0 100 \
    --comm-device gpu

