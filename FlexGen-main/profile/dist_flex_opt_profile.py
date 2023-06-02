import argparse
import json
import time
from itertools import count
import os
import pickle
import traceback
from typing import Union, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.dist_utils import initialize_distributed
from flexgen.flex_opt import (Policy, InputEmbed, OutputEmbed, SelfAttention,
                              MLP, TransformerLayer, OptLM, get_filename,
                              add_parser_arguments, get_test_inputs,
                              DUMMY_WEIGHT)
from flexgen.opt_config import get_opt_config
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, TorchTensor)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, array_4d, str2bool, project_decode_latency)
from torch import optim


#os.environ["NCCL_DEBUG"] = "TRACE"

"""
pipeline parallelism
"""
class DistOptLM(OptLM):

    def __init__(self, config, env, path, policy, pipeline_rank,
                 num_pipeline_stages, comm_device,num_pipeline_batches,max_new_tokens, num_inner_iterations=None,
                 async_comm=True,rank=None,):
        """
        The pipeline parallelism technique in dist_OPTLM divides the training process into multiple stages,
        where each stage consists of multiple pipeline steps. During each pipeline step, a portion of the input batch is processed by a specific pipeline stage,
        and the output is passed to the next pipeline stage for further processing.
        The pipeline steps are executed concurrently on different devices to increase the training throughput.

        The innerIteration parameter specifies the number of iterations to perform within a single pipeline stage before passing the output to the next stage.
        It essentially controls the granularity of the pipeline parallelism, determining how much computation is performed on each device before exchanging data between devices.

        config（example）
        name: str = "opt-125m"
        num_hidden_layers: int = 12
        max_seq_len: int = 2048
        hidden_size: int = 768
        n_head: int = 12
        input_dim: int = 768
        ffn_embed_dim: int = 3072
        pad: int = 1
        activation_fn: str = 'relu'
        vocab_size: int = 50272
        layer_norm_eps: float = 0.00001
        pad_token_id: int = 1
        dtype: type = np.float16
        """
        if rank is None:
            self.global_rank = args.rank
        else:
            self.global_rank = rank
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = self.policy.num_gpu_batches
        self.pipeline_rank = pipeline_rank
        # todo double check these equations
        self.pp_rank=pipeline_rank
        self.pipeline_group_size = num_pipeline_stages
        self.num_pipeline_stages = num_pipeline_stages
        self.num_inner_iterations = num_inner_iterations if num_inner_iterations is not None else num_pipeline_stages
        self.async_comm = async_comm
        self.seq_length=config.max_seq_len
        self.embedding_dim=config.ffn_embed_dim
        if comm_device == "cpu":
            self.comm_device = self.env.cpu
        elif comm_device == "gpu":
            self.comm_device = self.env.gpu
        else:
            raise ValueError(f"Invalid comm_device: {comm_device}")

        self.execute_gen_len = max_new_tokens

        print("global_rank"+str(self.global_rank))
        layers = []
        if pipeline_rank == 0:
            layers.append(InputEmbed(self.config, self.env, self.policy))
        # This block of code refers to stage-wise partitioning of layers of a neural network as described in the pipeline configuration.
        # pipeline_stage_sizes is a list that stores the number of layers that should be assigned to each stage in the pipeline.
        # The list is based on the number of hidden layers in the network represented by config.num_hidden_layers.
        pipeline_stage_sizes = [config.num_hidden_layers // num_pipeline_stages
                                + int(i < config.num_hidden_layers % num_pipeline_stages)
                                for i in range(num_pipeline_stages)]
        layer_start_ids = [0]
        for stage_size in pipeline_stage_sizes:
            layer_start_ids.append(layer_start_ids[-1] + stage_size)
            # range(num_pipeline_stages) produces values from 0 inclusive to num_pipeline_stages exclusive. This list of layer sizes is computed by dividing the number of layers by the number of pipeline stages and adding an offset of 1 where necessary. For example, if the number of layers is 12 and the number of pipeline stages is 3, then each stage should ideally have 4 layers. pipeline_stage_sizes would be computed as [4, 4, 4]. If the number of hidden layers is not a multiple of the number of pipeline stages, then the remaining layers are distributed amongst the first few stages and not the last stage. For instance,
            # if the number of layers is 13 and the number of pipeline stages is 3, then pipeline_stage_sizes would be [5, 4, 4].
        for i in range(layer_start_ids[pipeline_rank], layer_start_ids[pipeline_rank + 1]):
            if self.policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        if pipeline_rank == num_pipeline_stages - 1:
            layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        # It initializes CUDA streams used for loading weights, loading cache and storing cache with respect to training.
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        self.task = None
        self.init_all_weights()


        # todo added
        self.profiling_log = []
        self.global_profiling_log=[]
        self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.init_event.record()
        self.init_time_stamp = None
        self.init_time_stamp = time.time() * 1e+6
        self.dtype=np.float16
        # todo fix the bug
        self.micro_batch_num = self.num_pipeline_batches//self.num_inner_iterations
        self.micro_batch_size = self.num_inner_iterations
        self.enable_tidy_profiling = True
        # todo I don't know whether device is set to a right value
        self.device = self.comm_device
        self.num_pipeline_batches=num_pipeline_batches

        # It is a mechanism that allows for asynchronous execution and synchronization of CUDA streams.
        # A CUDA event marks a particular point in the CUDA stream, and other CUDA streams can wait for that event to occur before continuing with their own execution.
        # The torch.cuda.Event() constructor takes two optional arguments:
        # enable_timing (bool): If True, the event will record a timestamp when the event is reached.
        # blocking (bool): If True, a call to record() or synchronize() is a blocking operation that will wait for the event to complete. If False, the call to record() or synchronize() will return immediately, and the event will be completed asynchronously in the background.

        # [num_inner_iterations,execute_gen_len,num_layers,num_gpu_batches]
        self.compute_start_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.compute_ready_events =[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
        self.sending_future_t_i_j_k=[]
        self.receiving_future_t_i_j_k=[]
        self.send_start_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.send_ready_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.receive_start_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.receive_ready_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        #
        self.load_cache_ready_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.load_cache_start_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.store_cache_ready_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.store_cache_start_events = [[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]

        self.load_weight_start_events = [[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self.load_weight_ready_events = [[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self.store_hidden_start_events = [[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self.store_hidden_ready_events = [[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self.load_hidden_start_events=[[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self.load_hidden_ready_events=[[[[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                            for _ in range(self.num_gpu_batches)]
                                           for _ in range(self.num_layers)] for _ in range(self.execute_gen_len)]
                                         for _ in range(self.num_inner_iterations)]
                                         for _ in range(self.num_pipeline_batches // self.num_inner_iterations)]

        self._compute_micro_batch_size()

    def profile_mark_load_hidden_start(self,b, t, i, j, k):
        torch.cuda.current_stream().record_event(self.load_hidden_start_events[b][t][i][j][k])

    def profile_mark_load_hidden_ready(self,b,t,i,j,k):
        torch.cuda.current_stream().record_event(self.load_hidden_ready_events[b][t][i][j][k])

    def profile_mark_store_hidden_start(self,b, t, i, j, k):
        torch.cuda.current_stream().record_event(self.store_hidden_start_events[b][t][i][j][k])

    def profile_mark_store_hidden_ready(self, b, t, i, j, k):
        torch.cuda.current_stream().record_event(self.store_hidden_ready_events[b][t][i][j][k])


    def profile_mark_compute_start(self,t,i,j,k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        torch.cuda.current_stream().record_event(self.compute_start_events[t][i][j][k])

    def profile_mark_compute_end(self, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        torch.cuda.current_stream().record_event(self.compute_ready_events[t][i][j][k])

    def profile_mark_send_start(self, t,i,j, k):
        torch.cuda.current_stream().record_event(self.send_start_events[t][i][j][k])

    def profile_mark_send_ready(self, t, i, j, k):
        torch.cuda.current_stream().record_event(self.send_ready_events[t][i][j][k])

    def profile_mark_receive_start(self, t, i, j, k):
        torch.cuda.current_stream().record_event(self.receive_start_events[t][i][j][k])

    def profile_mark_receive_ready(self, t, i, j, k):
        torch.cuda.current_stream().record_event(self.receive_ready_events[t][i][j][k])



    def profile_mark_load_weight_start(self,b, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_weight_stream.record_event(self.load_weight_start_events[b][t][i][j][k])

    def profile_mark_load_weight_end(self,b,t,i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_weight_stream.record_event(self.load_weight_ready_events[b][t][i][j][k])

    def profile_mark_load_cache_start(self, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_cache_stream.record_event(self.load_cache_start_events[t][i][j][k])

    def profile_mark_load_cache_end(self, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")

        self.load_cache_stream.record_event(self.load_cache_ready_events[t][i][j][k])

    def profile_mark_store_cache_start(self, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_cache_stream.record_event(self.store_cache_start_events[t][i][j][k])

    def profile_mark_store_cache_end(self, t,i,j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.store_cache_ready_events[t][i][j][k])

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * self.seq_length * self.embedding_dim

        print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))

        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def load_weight(self, b, t, i, j, k):
        """
        @param b: batch index
        @param t: inner iteration index
        @param i: token index
        @param j: layer index
        @param k: GPU batch index
        @return:
        """
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            i = 0
            b += 1
        if b == self.num_pipeline_batches // self.num_inner_iterations:
            return

        # Load from weight_home to weight_read_buf
        with torch.cuda.stream(self.load_weight_stream):
            self.profile_mark_load_weight_start(b,t,i,j,k)
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
            self.profile_mark_load_weight_end(b,t,i,j,k)
            self.profiling_load_weight_stage(b,t,i,j,k)

    def init_cache(self, t, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[t][j][k])

    def load_cache(self, t, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            return

        # Load from cache_home to cache_read_buf
        with torch.cuda.stream(self.load_cache_stream):
            self.profile_mark_load_cache_start(t,i,j,k)
            self.layers[j].load_cache(self.cache_home[t][j][k], self.cache_read_buf[t][j][k], i)
            self.profile_mark_load_cache_end(t,i,j, k )
            self.profiling_load_cache_stage(t,i,j, k)

    def store_cache(self, t, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            t -= 1
        if t == -1:
            t = self.num_inner_iterations - 1
            i -= 1
        if i == -1:
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        with torch.cuda.stream(self.store_cache_stream):
            self.profile_mark_store_cache_start(t, i, j, k)
            self.layers[j].store_cache(self.cache_home[t][j][k], self.cache_write_buf[t][j][k], i)
            self.profile_mark_store_cache_end(t, i, j, k)
            self.profiling_store_cache_stage(t, i, j, k)

    def delete_cache(self, t, j, k):
        v = self.cache_home[t][j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, b, t, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            t += 1
        if t == self.num_inner_iterations:
            t = 0
            i += 1
        if i == self.execute_gen_len:
            i = 0
            b += 1
        if b == self.num_pipeline_batches // self.num_inner_iterations:
            return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        self.profile_mark_load_hidden_start(b,t,i,j,k)
        if j > 0: # load from the last layer
            val = self.hidden[t][i][j-1][k].pop().move(dst)
            self.hidden[t][i][j][k].store(val)
            self.profile_mark_load_hidden_ready(b, t, i, j, k)
            self.profiling_load_hidden_stage(b,t,i,j,k)
            return
        if self.num_pipeline_stages > 1 and not (i == 0 and self.pipeline_rank == 0):
            # Already received the input from previous hidden states
            self.hidden[t][i][j][k].val = self.hidden[t][i][j][k].val.move(dst)
            self.profile_mark_load_hidden_ready(b, t, i, j, k)
            self.profiling_load_hidden_stage(b, t, i, j, k)
            return
        gpu_batch_size = self.policy.gpu_batch_size
        num_gpu_batches = self.num_gpu_batches
        num_inner_iterations = self.num_inner_iterations
        left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
        right = left + gpu_batch_size
        if i == 0:  # load from the input ids
            val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int64)
            val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
        else:  # load from the last generated token
            pos = self.task.prompt_len + i
            val = dst.allocate((gpu_batch_size, 1), np.int64)
            val.load_from_np(self.output_ids[left:right, pos-1:pos])
        self.hidden[t][i][j][k].store(val)

        self.profile_mark_load_hidden_ready(b, t, i, j, k)
        self.profiling_load_hidden_stage(b, t, i, j, k)

    def store_hidden(self, b, t, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            t -= 1
        if t == -1:
            t = self.num_inner_iterations - 1
            i -= 1
        if i == -1:
            i = self.execute_gen_len - 1
            b -= 1
        if b == -1:
            return

        self.profile_mark_store_hidden_start(b,t,i,j,k)
        # Store to hidden states buffers
        if j != self.num_layers - 1 or self.pipeline_rank != self.num_pipeline_stages - 1 or i != self.execute_gen_len - 1:
            # Move to home
            x = self.hidden[t][i][j][k]
            if x.val:
                x.val = x.val.move(self.act_home)

        if j == self.num_layers - 1 and self.pipeline_rank == self.num_pipeline_stages - 1:
            # store to output
            if i == self.execute_gen_len - 1:  # last token
                hidden_val = self.hidden[t][i][j][k].pop()
            else:
                hidden_val = self.hidden[t][i][j][k].val

            ids = hidden_val.data.detach().cpu().numpy()
            gpu_batch_size = self.policy.gpu_batch_size
            num_gpu_batches = self.num_gpu_batches
            num_inner_iterations = self.num_inner_iterations
            left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
            right = left + gpu_batch_size
            pos = self.task.prompt_len + i
            self.output_ids[left:right, pos:pos+1] = ids
        self.profile_mark_store_hidden_ready(b,t,i,j,k)
        self.profiling_store_hidden_stage(b,t,i,j,k)
    # These two methods are used to send and receive tensors between pipeline stages using PyTorch's distributed package.
    # The goal is to send tensor values from a previous stage (sender rank) to the next stage (receiver rank),
    # a process known as inter-stage communication.
    def send_hidden(self, t, i, j, k, tag=0, async_=False):
        self.profile_mark_send_start(t,i,j,k)
        # Suppose we need to send tensors on GPUs
        x = self.hidden[t][i][j][k]
        # Firstly, the tensor is moved to the communication device (self.comm_device). This is because GPU tensors cannot be directly transmitted between different GPU ranks in PyTorch,
        # they are first required to be on the CPU.
        val = x.pop().move(self.comm_device)
        # The receiver rank is computed as the next pipeline rank ((self.pipeline_rank + 1) % self.num_pipeline_stages).
        receiver_rank = (self.pipeline_rank + 1) % self.num_pipeline_stages
        if async_:
            future = dist.isend(val.data, receiver_rank, tag=tag)
            # self.profile_mark_send_ready(t,i,j,k)
            # self.profiling_send_stage(t,i,j,k)
            self.sending_future_t_i_j_k.append([t,i,j,k])
            return future
        else:
            dist.send(val.data, receiver_rank, tag=tag)
            self.profile_mark_send_ready(t,i,j,k)
            self.profiling_send_stage(t,i,j,k)


    def recv_hidden(self, t, i, j, k, tag=0, async_=False):
        self.profile_mark_receive_start(t,i,j,k)
        sender_rank = (self.pipeline_rank - 1) % self.num_pipeline_stages
        val_holder = self.hidden[t][i][j][k]
        seq_len = self.task.prompt_len if i == 0 else 1
        shape, dtype = self.layers[j].input_act_shape_and_dtype(
            self.policy.gpu_batch_size, seq_len)
        # This method also pre-allocates a tensor of shape shape and data type dtype using self.comm_device, and places it in val_holder.val
        if val_holder.val is None:
            val_holder.val = self.comm_device.allocate(shape, dtype)
        else:
            val_holder.val = val_holder.val.move(self.comm_device)
        def move_value_callback():
            val_holder.val = val_holder.val.move(self.act_home)
        #  The move_value_callback callback function is provided so that once the tensor has been received, it is moved back to self.act_home to avoid occupying space on the communication device.
        if async_:
            # The method returns a future object that is completed once the message has been received.
            future = dist.irecv(val_holder.val.data, sender_rank, tag=tag)
            self.receiving_future_t_i_j_k.append([t,i,j,k])
            return future, move_value_callback
        else:
            dist.recv(val_holder.val.data, sender_rank, tag=tag)
            move_value_callback()
            self.profile_mark_receive_ready(t, i, j, k)
            self.profiling_receive_stage(t, i, j, k)

    def compute_layer(self, t, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.profile_mark_compute_start(t,i,j,k)
        self.layers[j].forward(self.hidden[t][i][j][k], self.cache_read_buf[t][j][k],
            self.weight_read_buf[j], self.attention_mask[t][k],
            self.cache_write_buf[t][j][k], i, k)
        self.profile_mark_compute_end(t,i,j,k)
        self.profiling_compute_stage(t,i,j,k)


    def update_attention_mask(self, b, t, i, k):
        """
        it is responsible for updating the attention mask for each token in the sequence during inference.
        @param b: index the pipeline micro batches
        @param t: index the inner iterations  t is the time index, or the index of the current generation step.
        The innerIteration variable class corresponds to the number of micro-batches processed per iteration within a single pipeline stage.
        ie:innerIteration means pipeline batch num and t mean the index of a single micro batch.
        @param i: index the tokens in the generated sequence
        @param k: GPU index
        @return:
        """
        #  It first checks if i > 0, which means that we are not processing the input prompt tokens. In this case, the attention mask has already been computed for the previous token, and we just need to extend the existing mask to cover the current token.
        if i > 0:
            mask = self.attention_mask[t][k]
            assert mask.val is not None
            # 这里的device已经在之前指定好了，然后extend_attention_mask即是计算
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return
        # Otherwise, if i == 0, we are processing the input prompt tokens, and we need to compute the attention mask for the entire sequence.
        # The function first extracts the input prompt tokens from the output IDs tensor using the left and right indices.
        # It then creates a binary mask tensor by checking which tokens in the prompt are not equal to the padding token ID.
        # Finally, it stores the attention mask tensor in the self.
        # attention_mask buffer at the specified t and k indices.
        gpu_batch_size = self.policy.gpu_batch_size
        num_gpu_batches = self.num_gpu_batches
        num_inner_iterations = self.num_inner_iterations
        # gpu_batch_size is the size of a single GPU batch, which determines the number of tokens processed by each GPU.
        # num_gpu_batches is the number of GPU batches(maybe stand for the number of gpu?). k is the index of the current GPU batch.
        # num_gpu_batches corresponds to the number of GPU devices used for processing the micro-batches.
        left = ((b * num_inner_iterations + t) * num_gpu_batches + k) * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[t][k].val = val



    def get_profiling_log(self):
        return self.profiling_log

    def get_global_profiling_log(self):
        return self.global_profiling_log

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        # cut_gen_len: an optional integer representing the maximum length of generated text to return.
        # If specified, the generated text is truncated to this length.
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        assert stop is None, "Not implemented."
        num_pipeline_stages = self.num_pipeline_stages
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        num_prompts = len(task.inputs)
        num_inner_iterations = self.num_inner_iterations

        assert num_prompts % (gpu_batch_size * num_gpu_batches) == 0
        num_pipeline_batches = num_prompts // (gpu_batch_size * num_gpu_batches)
        self.num_pipeline_batches = num_pipeline_batches
        assert num_pipeline_batches % num_inner_iterations == 0
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        # In this code block, the generate function initializes an output_ids array of shape (num_prompts, prompt_len + gen_len) and data type np.int64. This array will hold the final output of the generation process.
        # The num_prompts variable represents the number of input prompts given to the function. prompt_len is the length of each prompt and gen_len is the maximum length of the generated text.
        # The code sets all values in the output_ids array to 1 using the np.ones function, and then copies the input prompts into the first prompt_len columns of each row of the output_ids array. This ensures that the generated text will start from the given input prompt.
        # Overall, this code block sets up the output_ids array to store the generated text, and initializes it with the input prompts.
        self.output_ids = np.ones((num_prompts, prompt_len + gen_len), dtype=np.int64)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch, t-th stage.
        # cache[t][j][k]
        self.cache_home = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_3d(num_inner_iterations, num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # hidden[t][i][j][k]
        self.hidden = array_4d(num_inner_iterations, gen_len, num_layers, num_gpu_batches, ValueHolder)
        # attention_mask[t][k]
        # In the case of attention_mask, it doesn't need an i dimension because the attention mask is the same for all tokens in a given input sequence.
        # It is a binary mask that indicates which tokens should be attended to and which tokens should be ignored. Therefore, it is only necessary to specify the mask for each input sequence (k dimension) and for each generation stage (t dimension).
        self.attention_mask = array_2d(num_inner_iterations, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for t in range(num_inner_iterations):
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.init_cache(t, j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        dist.barrier()

        # Generate

        self.profiling_log = []
        if not overlap:
            # No overlap
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if self.policy.num_gpu_batches == 1:
                self.generation_loop_overlap_one_batch()
            else:
                self.generation_loop_overlap_multi_batch()

        # Delete cache
        for t in range(num_inner_iterations):
            for j in range(num_layers):
                for k in range(num_gpu_batches):
                    self.delete_cache(t, j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()


        trace_file = "prefilling/" + "generate_overlap_" + str(args.overlap)+"num_gpu_batches_"+str(self.policy.num_gpu_batches) + "_percent_" + str(
            args.percent) + "pp_rank:" + str(self.pp_rank) + \
                     '.json'

        print("output:" + trace_file)
        with open(trace_file, 'w') as outfile:
            json.dump(self.profiling_log, outfile)





        return self.output_ids

    def send_recv_hidden(self, sending_job, receiving_job):
        """
        The method takes two arguments, sending_job and receiving_job. Both arguments are tuples with two elements.
        The first element is the tensor that will be sent or received. The second element is an integer indicating the index of the tensor in a larger sequence of tensors.
        If a job is not provided, then the corresponding variables st, si, rt, ri are set to None.
        t stands for the index of inner_iterations
        i stands for the index of generated token
        @return:
        """
        st, si = sending_job if sending_job is not None else (None, None)
        rt, ri = receiving_job if receiving_job is not None else (None, None)
        # In order to determine whether a tensor should be sent or received, the code sets the variables sending and receiving based on whether the sending_job and receiving_job are not None.
        # Additionally, if a sending job is set, the code looks at the self.execute_gen_len and self.pipeline_rank variables to determine whether it is appropriate to send the message.
        # Similarly, if a receiving job is set, the code looks at the ri and self.pipeline_rank variable to determine whether it is appropriate to receive the message.
        sending = sending_job is not None and not (si == self.execute_gen_len - 1 and self.pipeline_rank == self.num_pipeline_stages - 1)
        receiving = receiving_job is not None and not (ri == 0 and self.pipeline_rank == 0)

        def _send():
            sending_futures = []

            if not sending:
                return sending_futures
            for k in range(self.num_gpu_batches):
                sending_future = self.send_hidden(st, si, self.num_layers - 1, k, self.sending_tag, async_=self.async_comm)
                sending_futures.append(sending_future)
                self.sending_tag += 1
            return sending_futures

        def _recv():
            receiving_futures = []
            if not receiving:
                return receiving_futures
            for k in range(self.num_gpu_batches):

                receiving_future = self.recv_hidden(rt, ri, 0, k, self.receiving_tag, async_=self.async_comm)
                receiving_futures.append(receiving_future)
                self.receiving_tag += 1
            return receiving_futures

        # The order of sending and receiving is carefully structured to prevent deadlock.
        # If the self.pipeline_rank is 0, then the nodes receive first and send next. Otherwise, they send first and then receive.
        # Use special order below to avoid deadlock


        if self.pipeline_rank == 0:
            # Receive first and then send
            receiving_futures = _recv()
            sending_futures = _send()

        else:
            # Send first and then receive
            sending_futures = _send()
            receiving_futures = _recv()
        # Finally, the method waits for the asynchronous communication to complete and returns the sending_futures and receiving_futures.
        # If the communication is asynchronous,
        # it also waits for the futures to complete and triggers the specified callback functions when the receive call finishes.
        if self.async_comm:
            sending_index = 0
            receiving_index = 0
            for sending_future in sending_futures:
                sending_future.wait()
                t_i_j_k_item=self.sending_future_t_i_j_k[sending_index]
                self.profile_mark_send_ready(t_i_j_k_item[0],t_i_j_k_item[1],t_i_j_k_item[2],t_i_j_k_item[3])
                self.profiling_send_stage(t_i_j_k_item[0],t_i_j_k_item[1],t_i_j_k_item[2],t_i_j_k_item[3])
                sending_index+=1

            for receiving_future, callback in receiving_futures:
                receiving_future.wait()
                callback()
                t_i_j_k_item=self.receiving_future_t_i_j_k[receiving_index]
                self.profile_mark_receive_ready(t_i_j_k_item[0],t_i_j_k_item[1],t_i_j_k_item[2],t_i_j_k_item[3])
                self.profiling_receive_stage(t_i_j_k_item[0],t_i_j_k_item[1],t_i_j_k_item[2],t_i_j_k_item[3])
                receiving_index+=1


    def generation_loop_normal(self):
        """
        This code defines a loop for generating text using the GPT model. The loop iterates over a number of batches (num_pipeline_batches) and for each batch it generates text of a specific length (execute_gen_len).
        Within each batch, the loop iterates over the inner iterations (num_inner_iterations). For each inner iteration, the loop generates one token at a time (i), and performs the following steps:
        For each GPU batch (num_gpu_batches), it updates the attention mask using update_attention_mask().
        If there is more than one pipeline stage (num_pipeline_stages > 1), it sends and receives hidden states using send_recv_hidden().
        For each layer (num_layers) of the model, it loads the weights using load_weight(), and then synchronizes the processes using sync().
        For each GPU batch, it loads the cache and hidden states using load_cache() and load_hidden(), respectively. It then synchronizes the processes.
        It computes the output of the layer using compute_layer(), and synchronizes the processes.
        It stores the updated hidden states and cache using store_hidden() and store_cache(), respectively, and synchronizes the processes.
        Finally, it updates the last_sending_job variable with the current token and inner iteration.
        If there is more than one pipeline stage, the loop sends and receives hidden states between pipeline stages using send_recv_hidden(). Finally, the loop synchronizes the processes using dist.barrier().
        """
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None
        # b is used to index the pipeline batches, t is used to index the inner iterations,
        # i is used to index the tokens in the generated sequence, and k is used to index the GPUs.
        # In pipeline parallelism, the input data is split into smaller portions called micro-batches, which are then processed in parallel across multiple stages of the pipeline.
        # The innerIteration variable class corresponds to the number of micro-batches processed per iteration within a single pipeline stage.
        # Its value thus represents the degree of parallelism between micro-batches within a stage,
        # and influences the overall performance of the pipeline.
        # num_inner_iterations MEAN micro_batch_num
        # b means the the index of micro_batch
        # b 指的是某个阶段的 pipeline 中，该进程负责完成的子批次的数量。每个子批次通常包含多个样本，且它们会被分散到多个 GPU 上进行计算。
        # t 指的是在 "generate" 或者 "generate-prompt" 步骤中的第几个小批次（micro-batch）。即the index of micro_batches
        # k 是指在当前小批次 t 中，进程应该计算的样本数量。
        # 先循环i再循环t是因为 产生每个mirco_batch同一sequenceid位置上的结果
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    for k in range(self.num_gpu_batches):

                        self.update_attention_mask(b, t, i, k)
                    # If there is more than one pipeline stage (num_pipeline_stages > 1), it sends and receives hidden states using send_recv_hidden().
                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):
                        for k in range(self.num_gpu_batches):

                            self.load_weight(b, t, i, j, k)


                        self.sync()
                        # For each GPU batch, it loads the cache and hidden states using load_cache() and load_hidden(), respectively. It then synchronizes the processes.

                        for k in range(self.num_gpu_batches):
                            # For each GPU , it loads the cache and hidden states using load_cache() and load_hidden(), respectively. It then synchronizes the processes.

                            self.load_cache(t, i, j, k)


                            self.load_hidden(b, t, i, j, k)

                            self.sync()
                            # It computes the output of the layer using compute_layer(), and synchronizes the processes.
                           # self.torch_comp_stream.wait_event(self.load_cache_ready_events[k])

                            self.compute_layer(t, i, j, k)

                            #self.torch_comp_stream.record_event(self.forward_comp_ready_events[k])
                            self.sync()
                            # It stores the updated hidden states and cache using store_hidden() and store_cache(), respectively, and synchronizes the processes.
                            self.store_hidden(b, t, i, j, k)

                            self.store_cache(t, i, j, k)


                            self.sync()
                    # Finally, it updates the last_sending_job variable with the current token and inner iteration.
                    last_sending_job = (t, i)

                    timers(timer_name).stop()
         #  if self.enable_tidy_profiling:
         #   self.profiling_forward_stage()


        # If there is more than one pipeline stage, the loop sends and receives hidden states between pipeline stages using send_recv_hidden().
        # Finally, the loop synchronizes the processes using dist.barrier().
        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            dist.barrier()

    def profiling_load_weight_stage(self,b,t,i,j,k):
        torch.cuda.synchronize()

        load_weight_slot = self.load_weight_start_events[b][t][i][j][k].elapsed_time(self.load_weight_ready_events[b][t][i][j][k]) * 1e+3
        load_weight_log = {"name": "load_weight", "ph": "X", "pid": self.pp_rank, "tid": "load-weight",
                     "dur": load_weight_slot,"ts": self.get_ts(self.load_weight_start_events[b][t][i][j][k]),
                     "args": {"micro-batch-index": b,"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "good"}
        self.profiling_log.append(load_weight_log)

    def profiling_load_cache_stage(self,t,i,j,k):
        torch.cuda.synchronize()

        load_cache_slot = self.load_cache_start_events[t][i][j][k].elapsed_time(self.load_cache_ready_events[t][i][j][k]) * 1e+3
        load_cache_log = {"name": "load_cache", "ph": "X", "pid": self.pp_rank, "tid": "load-cache",
                     "dur": load_cache_slot,"ts": self.get_ts(self.load_cache_start_events[t][i][j][k]),
                     "args": {"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "cq_build_running"}
        self.profiling_log.append(load_cache_log)

    def profiling_store_cache_stage(self,t, i, j, k):
        torch.cuda.synchronize()

        store_cache_slot = self.store_cache_start_events[t][i][j][k].elapsed_time(self.store_cache_ready_events[t][i][j][k]) * 1e+3
        store_cache_log = {"name": "store_cache", "ph": "X", "pid": self.pp_rank, "tid": "store-cache",
                     "dur": store_cache_slot,"ts": self.get_ts(self.store_cache_start_events[t][i][j][k]),
                     "args": {"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "cq_build_running"}
        self.profiling_log.append(store_cache_log)

    def profiling_compute_stage(self,t,i,j,k):
        torch.cuda.synchronize()

        compute_slot = self.compute_start_events[t][i][j][k].elapsed_time(self.compute_ready_events[t][i][j][k]) * 1e+3
        compute_log = {"name": "compute", "ph": "X", "pid": self.pp_rank, "tid": "compute",
                     "dur": compute_slot,"ts": self.get_ts(self.compute_start_events[t][i][j][k]),
                     "args": {"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "black"}

        self.profiling_log.append(compute_log)

    def profiling_send_stage(self,t,i,j,k):
        torch.cuda.synchronize()

        send_slot = self.send_start_events[t][i][j][k].elapsed_time(self.send_ready_events[t][i][j][k]) * 1e+3
        send_log = {"name": "send", "ph": "X", "pid": self.pp_rank, "tid": "send",
                     "dur": send_slot,"ts": self.get_ts(self.send_start_events[t][i][j][k]),
                     "args": {"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "olive"}

        self.profiling_log.append(send_log)

    def profiling_receive_stage(self,t,i,j,k):
        torch.cuda.synchronize()

        receive_slot = self.receive_start_events[t][i][j][k].elapsed_time(self.receive_ready_events[t][i][j][k]) * 1e+3
        receive_log = {"name": "receive", "ph": "X", "pid": self.pp_rank, "tid": "receive",
                     "dur": receive_slot,"ts": self.get_ts(self.receive_start_events[t][i][j][k]),
                     "args": {"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "olive"}

        self.profiling_log.append(receive_log)


    def profiling_load_hidden_stage(self,b, t, i, j, k):
        torch.cuda.synchronize()

        slot = self.load_hidden_start_events[b][t][i][j][k].elapsed_time(self.load_hidden_ready_events[b][t][i][j][k]) * 1e+3
        log = {"name": "load-hidden", "ph": "X", "pid": self.pp_rank, "tid": "load-hidden",
                       "dur": slot, "ts": self.get_ts(self.load_hidden_start_events[b][t][i][j][k]),
                       "args": {"micro-batch-index": b,"inner-micro-batch-index":t,"generate-token-index":i,"layer-index":j, "gpu-index":k}, "cname": "rail_response"}

        self.profiling_log.append(log)

    def profiling_store_hidden_stage(self, b, t, i, j, k):
        torch.cuda.synchronize()

        slot = self.store_hidden_start_events[b][t][i][j][k].elapsed_time(
            self.store_hidden_ready_events[b][t][i][j][k]) * 1e+3
        log = {"name": "store-hidden", "ph": "X", "pid": self.pp_rank, "tid": "store-hidden",
               "dur": slot, "ts": self.get_ts(self.load_hidden_start_events[b][t][i][j][k]),
               "args": {"micro-batch-index": b, "inner-micro-batch-index": t, "generate-token-index": i, "layer-index": j,
                        "gpu-index": k}, "cname": "rail_response"}

        self.profiling_log.append(log)


    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def generation_loop_overlap_one_batch(self):
        assert self.num_gpu_batches == 1
        # Prologue
        self.load_weight(0, 0, 0, 0, 0)
        self.sync()
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        # Generate
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    self.update_attention_mask(b, t, i, 0)

                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):

                        self.load_weight(b, t, i, j+1, 0)



                        self.load_cache(t, i, j+1, 0)


                        self.load_hidden(b, t, i, j, 0)


                        self.compute_layer(t, i, j, 0)



                        self.store_cache(t, i, j-1, 0)


                        self.store_hidden(b, t, i, j, 0)
                        self.sync()

                    last_sending_job = (t, i)

                    timers(timer_name).stop()

        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            dist.barrier()

    def generation_loop_overlap_multi_batch(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None

        print("num_pipeline_batches in multi_batch")
        print(self.num_pipeline_batches)
        print("num_inner_iterations in multi_batch")
        print(self.num_inner_iterations)
        # prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, 0, 0, k)
        # num_pipeline_batches // num_inner_iterations refers to the effective batch size used for updating the optimizer parameters in pipeline parallelism when using an iterative optimization algorithm such as stochastic gradient descent (SGD). It is computed by dividing the number of pipeline batches by the number of inner iterations.
        # For example, if num_pipeline_batches is set to 16 and num_inner_iterations is set to 4, the effective batch size for each update is 16 // 4 = 4. This means that the optimizer parameters are updated every 4 mini-batches, with each update using an effective batch size of 4.
        for b in range(self.num_pipeline_batches // self.num_inner_iterations):
            for i in range(self.execute_gen_len):
                for t in range(self.num_inner_iterations):
                    timer_name = "generate-prompt" if i == 0 else "generate"
                    timers(timer_name).start()
                    for k in range(self.num_gpu_batches):
                        self.update_attention_mask(b, t, i, k)

                    if self.num_pipeline_stages > 1:
                        self.send_recv_hidden(last_sending_job, (t, i))

                    for j in range(self.num_layers):
                        for k in range(self.num_gpu_batches):

                            self.load_weight(b, t, i, j + 1, k)
                            self.load_cache(t, i, j, k + 1)
                            self.load_hidden(b, t, i, j, k)
                            self.compute_layer(t, i, j, k)
                            self.store_cache(t, i, j, k - 1)
                            self.store_hidden(b, t, i, j, k)
                            self.sync()

                    last_sending_job = (t, i)

                    timers(timer_name).stop()

        if self.num_pipeline_stages > 1:
            self.send_recv_hidden(last_sending_job, None)
            dist.barrier()




def comm_test(comm_device):
    # A small all_reduce for warmup.
    a = torch.ones(1).to(comm_device)
    dist.all_reduce(a)
    assert a.item() == args.world_size

def run_flexgen_dist(args):
    t_name = args.model.replace("175b", "66b")
    tokenizer = AutoTokenizer.from_pretrained(t_name, padding_side="left")
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.world_size
    num_prompts = args.num_gpu_batches * args.gpu_batch_size * num_inner_iterations * 1
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)

    comm_test(gpu.dev if args.comm_device == "gpu" else cpu.dev)

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    num_pipeline_batches = num_prompts // (args.gpu_batch_size * args.num_gpu_batches)
    model = DistOptLM(opt_config, env, args.path, policy, args.rank,
                      args.world_size, args.comm_device,num_pipeline_batches=num_pipeline_batches,max_new_tokens=gen_len, num_inner_iterations=num_inner_iterations,
                      async_comm=args.async_comm)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=2, verbose=args.verbose)

        # global_profiling_log = model.get_global_profiling_log()
        # trace_file = "prefilling/" + "warm_up_overlap_" + str(args.overlap) + "_percent_" + str(args.percent)+"pp_rank"+model.pp_rank + \
        #              '.json'
        # print("output:" + trace_file)
        #
        # with open(trace_file, 'w') as outfile:
        #     json.dump(global_profiling_log, outfile)

        print("benchmark - generate")
        for timer_name in ["generate-prompt", "generate"]:
            timers(timer_name).reset()

        # model.global_profiling_log=[]

        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        prompt_costs = timers("generate-prompt").costs
        generate_costs = timers("generate").costs


    finally:
        env.close_copy_threads()

    if args.rank != args.world_size - 1:
        return

    # Log output
    prefill_latency = sum(prompt_costs)
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        costs = np.array(generate_costs).reshape(-1, cut_gen_len-1).sum(axis=0).tolist()
        print("costs" + str(costs))
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
        print("decode_latency" + str(decode_latency))
    else:
        decode_latency = sum(generate_costs)
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    print("num_prompts"+str(num_prompts))
    print("gen_len"+str(gen_len))

    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = args.debug_mode or cut_gen_len

    log_str = (f"model size: {opt_config.model_bytes()/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (prefill): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"prefill latency: {prefill_latency:.2f} s\t"
               f"prefill throughput: {prefill_throughput:.2f} token/s\n"
               f"decode latency: {decode_latency:.2f} s\t"
               f"decode throughput: {decode_throughput:.2f} token/s\n"
               f"total latency: {total_latency:.2f} s\t"
               f"total throughput: {total_throughput:.2f} token/s")
    print(log_str)

    if not args.no_log:
        if args.log_file == "auto":
            basename = f"rank-{args.rank}-{get_filename(args)}"
            log_filename = basename + ".log"
        else:
            log_filename = args.log_file
        with open(log_filename, "a") as fout:
            fout.write(log_str + "\n")





def add_distributed_parser_arguments(parser):
    parser.add_argument('--head-ip', type=str, default=None, help='the IP address of the head node')
    parser.add_argument('--port', type=int, default=None, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=None)
    parser.add_argument('--local-rank', metavar='I', type=int, default=None)
    parser.add_argument('--world-size', metavar='N', type=int, default=None)
    parser.add_argument('--use-mpi', action='store_true', default=False,
                        help="Get distributed info from MPI")
    parser.add_argument('--comm-device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='communication through gpu nvlink or cpu memory '
                             'and socket')
    parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    parser.add_argument('--async-comm', action='store_true', default=False,
                        help="Use asynchronous communication")

def get_learning_arguments_str(policy):
    return '_b' + str(args.batch_size) + '_' + str(policy.gpu_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()
    print("test")
    if args.head_ip is not None and args.port is not None:
        if args.use_mpi:
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        initialize_distributed(args.head_ip, args.port, args.world_size,
                               args.rank, args.local_rank, args.comm_device)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    assert len(args.percent) == 6



    try:
        run_flexgen_dist(args)


        # print()

    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise e

