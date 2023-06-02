"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0

refactor problems:
1.the operations on buffer Array(elements are ValueHolder which points to TorchTensor, tuple and so on) are diffcult to control-- mayBe wan extract some function further;
2.maybe we can extract super class for layers

General understandings:
1.Weights and cache perform different functions in a neural network, and as such, have different operations for loading, storing, and initializing.
Weight:
Initialization: Weight are initialized during model creation and training. The weight initialization process for each layer typically involves specifying the shape, dtype, and other necessary parameters for each weight tensor.
Loading: The weights are loaded from pre-trained model checkpoints or saved model files before the model is used for inference or fine-tuning. The loading process updates the existing model weights and overwrites any previous values.
Storing: The weights can be saved to a model checkpoint or saved model file after training or fine-tuning. This allows the model to be reloaded with the same weights later.
Cache:
Initialization: The cache for each layer must be initialized before training or inference. This initialization process involves allocating memory for the cache on the appropriate device (GPU, CPU, or mixed) based on the policy.
Loading: The cache is loaded from the appropriate device, and a copy is made to the working buffer for use in the current inference batch. The loading process can be affected by compression and sparsity policies.
Storing: The cache is updated and stored after each inference batch. The updated cache can then be used in the next inference batch.
2.The layer of embedding, MLP, self-Attention, TransformerLayer(combination of MLP and self-Attention) can customized for the need of OPT
while using TorchDevice to calculate, allocate spaces for buffer arrays in different devices.

"""

import argparse
import dataclasses
import json
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
# This code defines a Policy data class which specifies various parameters for the training policy, such as the batch size, the percentage of computation to be done on CPU and GPU, whether to use pinned memory for weights on CPU, etc.
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def init_weight_list(weight_specs, policy, env):
    """
    This is a function called init_weight_list that initializes a list of weights based on the given weight_specs (a list of tuples containing the shape, dtype, and filename of each weight) and the given policy and env.
    """
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]
    # for each weight, it calculates the midpoint of the weight's size in terms of the cumulative sum of all weight sizes,
    # and assigns the weight to a device based on the percentage of each device's storage capacity and the midpoint of the weight's size.

    # This approach is used to balance the distribution of weights across available devices based on their storage capacity.
    # By calculating the midpoint of each weight's size in terms of the cumulative sum of all weight sizes,
    # the function tries to assign larger weights to devices with more storage capacity and smaller weights to devices with less storage capacity.

    # The function then assigns each weight to a device based on the percentage of each device's storage capacity and the midpoint of the weight's size.
    # This ensures that weights are distributed evenly across available devices, which can improve the training performance and reduce the memory footprint of the model.

    # Overall, this approach optimizes the use of available resources, which can be critical in large-scale deep learning systems where hardware resources are limited and expensive.
    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename = weight_specs[i]
        # todo what is pin_memory?
        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight
        # If the weight's shape is less than two dimensions (i.e., it is a scalar or vector), it is pinned to memory and not compressed.
        # Otherwise, the compress flag in the policy is checked to determine whether to compress the weight or not.
        # If compress is False, the weight is allocated on the chosen device and loaded from the specified file.
        # If the filename contains the string "DUMMY_WEIGHT", the weight is initialized to an array of ones.
        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                weight.load_from_np(np.ones(shape, dtype))
                #weight.load_from_np(np.random.rand(*shape).astype(dtype))
        # If compress is True, the weight is allocated as a compressed tensor on the chosen device using the compressed_device method of the device,
        # and is loaded from the specified file or initialized to an array of ones for dummy weights.
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(weight_specs[i][2])
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))

        ret.append(weight)
    return ret


class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        """
        The init_weight method initializes the weight information and stores it in the weight_home attribute.
        It creates a list of weight_specs which define the shape, data type, and file name of each weight.
        The list has two elements for w_token and w_pos, which represent the weights for the input tokens and the input positions.
        @param weight_home:
        @param path:
        """
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        # The purpose of having the weight_home attribute is to keep the weights in memory on a specific device, rather than having them moving back and forth between devices.
        # This can improve the performance of the model since moving data between devices can be expensive and can cause delays during training.
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        """
        weight_read_buf is the buffer used to read weights from weight_home. Since the weights are kept on a specific device, they need to be read into all the GPU devices used for parallel training.
        This is done by first copying the weights on the CPU node or storage, followed by copying them onto weight_read_buf, which is accessed by the parallel GPUs.
        weight_write_buf, on the other hand, is used to write the weights that are updated during the forward or backward pass of each GPU during parallel training.
        The write buffer stores the newly computed weights from each GPU and is merged by the DeviceTensorList method after the forward or backward pass from each GPU.

        In summary, while weight_home holds the weights on a specific device, weight_read_buf and weight_write_buf hold the weights for parallel training by different GPUs,
        with weight_read_buf reading the weights into GPU devices and weight_write_buf writing newly updated weights after each parallelized training process iteration.
        """
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        """
        The forward() method is the primary method of the class.
        The purpose of this method is to compute the input embedding and write it back to the hidden state.
        This is done by extracting weight information from weight_read_buf. The compute.opt_input_embed() method is then called to calculate
        """
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        # it gets the attention mask mask from attention_mask and copies it to self.compute using smart_copy().
        # It also extracts the weights w_token and w_pos from weight_read_buf.
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        # If k is equal to self.policy.num_gpu_batches - 1, it means that this is the last GPU batch, and so the function pops the weights from weight_read_buf.
        # Otherwise, it gets the weights from weight_read_buf.val.
        # why there is the operation of poping? Because of zig-zag policy, we are turning to next layer, so we need to load weight again.
        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        # Finally, it calls self.compute.opt_input_embed() to calculate the input embeddings. This method takes the hidden state h, the attention mask mask, the weights w_token and w_pos, and the padding token ID self.config.pad_token_id as inputs.
        # It also takes donate as another input argument to indicate whether to donate memory. The resulting input embeddings are stored in hidden.val.
        h = self.compute.opt_input_embed(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, b_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h


class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        # The function first checks the cache_gpu_percent, cache_cpu_percent, and cache_disk_percent properties of self.policy.
        # Depending on the value of these properties, the function assigns a device object to a particular device (either GPU, CPU, disk, or mixed).
        # After determining the device, the function checks if self.policy.compress_cache is set to True.
        # If so, it ensures that the device is not a mixed device since compression is not supported on mixed devices.
        # If necessary, the function updates the device to the compressed device.
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        # TorchTensor.
        # Finally, the function calls the init_cache_one_gpu_batch method of device with three arguments (self.config, self.task, self.policy) to create a cache.
        # This cache is then stored in cache_home using the store method.
        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            # the implementation of disabling the computation delegation
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h


class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, k):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, k)
        self.mlp.load_weight(home2, read_buf2, k)
        if k == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_gpu_batch(self, cache_home):
        self.attention.init_cache_one_gpu_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        if k == self.policy.num_gpu_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, k)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, k)


# 这段代码定义了一个 OptLM 类，该类用于实现基于PyTorch的优化器，用于处理语言模型任务。
# 在 __init__ 函数中，首先判断传入的 config 参数是否为字符串类型，如果是则调用 get_opt_config 函数获取配置信息，如果不是则将传入的 config 直接赋值给 self.config。
# 接着将传入的 env、path 和 policy 分别赋值给 self.env、self.path 和 self.policy，将 policy.num_gpu_batches 赋值给 self.num_gpu_batches。
# 然后定义了一个列表 layers，并将 InputEmbed、SelfAttention、MLP、TransformerLayer 和 OutputEmbed 这些类的实例添加到列表中。
# 这些类用于实现不同的神经网络层，用于对输入进行编码和处理。
# 接下来，判断 policy 中的 act_gpu_percent、act_cpu_percent 和 act_disk_percent 参数，
# 根据其取值将 self.act_home 设置为对应的环境对象（env）中的 GPU、CPU 或磁盘。
# 定义了三个 CUDA 流（self.load_weight_stream、self.load_cache_stream 和 self.store_cache_stream），用于异步地在 GPU 上加载权重和缓存。
# 接着定义了多个值为 ValueHolder 的数组，这些数组用于存储中间结果。其中，self.cache_home、self.cache_read_buf 和 self.cache_write_buf 分别表示缓存的主数组、读取缓存的缓冲区和写入缓存的缓冲区，self.weight_read_buf 表示权重读取缓冲区，self.attention_mask 表示注意力掩码。

# 最后调用 init_all_weights 函数，用于初始化所有的权重。
class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy,
                 gen_len: int,
                 cut_gen_len: Optional[int] = None,
    ):
        """
        The OptLM class also sets up CUDA streams and intermediate tensors used for computation. These include buffers for storing values for each token, layer, and GPU batch, as well as a buffer for reading the weight values and an array for storing the attention masks.
        The init_all_weights method initializes all the weight values used in the network.
        """
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
                # OutputEmbed是最后一个线性层，用于输出结果
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
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        # The cache variable is a 2-dimensional array with num_layers rows and num_gpu_batches columns,
        # where num_layers is the total number of layers in the neural network
        # and num_gpu_batches is the number of batches that can be processed simultaneously on the GPU.

        # Each element of the cache array (cache[j][k]) represents the intermediate computation results (cache) for the j-th layer and the k-th GPU batch.
        # Since the size of the intermediate results varies depending on the layer and the input size,
        # the cache variable needs to be a 2D array to store intermediate results for all layers and all batches.
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

        # todo added
        self.profiling_log = []
        self.global_profiling_log = []
        self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.init_event.record()
        self.init_time_stamp = None
        self.init_time_stamp = time.time() * 1e+6
        self.dtype = np.float16
        self.execute_gen_len = gen_len

        self.enable_tidy_profiling = True
        #
        # self.device = self.comm_device

        # It is a mechanism that allows for asynchronous execution and synchronization of CUDA streams.
        # A CUDA event marks a particular point in the CUDA stream, and other CUDA streams can wait for that event to occur before continuing with their own execution.
        # The torch.cuda.Event() constructor takes two optional arguments:
        # enable_timing (bool): If True, the event will record a timestamp when the event is reached.
        # blocking (bool): If True, a call to record() or synchronize() is a blocking operation that will wait for the event to complete. If False, the call to record() or synchronize() will return immediately, and the event will be completed asynchronously in the background.


        # [micro_batch_num,num_gpu_batch]
        self.compute_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]
        self.compute_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]
        # [micro_batch_num,num_gpu_batch]
        self.load_cache_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]

        self.load_cache_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]

        self.store_cache_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]

        self.store_cache_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]

        self.load_weight_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]
        print("self.load_weight_start_events ")
        my_array = np.array(self.load_weight_start_events)
        print(my_array.shape)

        self.load_weight_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]

        self.load_hidden_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                     in range(self.execute_gen_len)]
        self.load_hidden_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                      in range(self.execute_gen_len)]
        self.store_hidden_start_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                      in range(self.execute_gen_len)]
        self.store_hidden_ready_events = [[[torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                      for _ in range(self.num_gpu_batches)] for _ in range(self.num_layers)] for _
                                      in range(self.execute_gen_len)]

    def profile_mark_compute_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        torch.cuda.current_stream().record_event(self.compute_start_events[i][j][k])

    def profile_mark_compute_end(self,i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        torch.cuda.current_stream().record_event(self.compute_ready_events[i][j][k])

    def profile_mark_load_weight_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_weight_stream.record_event(self.load_weight_start_events[i][j][k])

    def profile_mark_load_weight_end(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_weight_stream.record_event(self.load_weight_ready_events[i][j][k])

    def profile_mark_load_cache_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_cache_stream.record_event(self.load_cache_start_events[i][j][k])

    def profile_mark_load_cache_end(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")

        self.load_cache_stream.record_event(self.load_cache_ready_events[i][j][k])

    def profile_mark_store_cache_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_start")
        self.load_cache_stream.record_event(self.store_cache_start_events[i][j][k])

    def profile_mark_store_cache_end(self, i, j , k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.store_cache_ready_events[i][j][k])

    def profile_mark_load_hidden_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.load_hidden_start_events[i][j][k])

    def profile_mark_load_hidden_end(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.load_hidden_ready_events[i][j][k])

    def profile_mark_store_hidden_start(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.store_hidden_start_events[i][j][k])

    def profile_mark_store_hidden_end(self, i, j, k):
        # print("record micro_batch"+str(t)+" gpu "+str(k)+"load_cache_end")
        self.load_cache_stream.record_event(self.store_hidden_ready_events[i][j][k])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3


    def profiling_load_weight_stage(self, i , j , k):
        torch.cuda.synchronize()

        load_weight_slot = self.load_weight_start_events[i][j][k].elapsed_time(self.load_weight_ready_events[i][j][k]) * 1e+3
        load_weight_log = {"name": "load_weight", "ph": "X", "pid": 0, "tid": "1. load-weight",
                     "dur": load_weight_slot,"ts": self.get_ts(self.load_weight_start_events[i][j][k]),
                     "args": {"gen-len-id": i,"layers-id": j,"gpu-id": k}, "cname": "good"}
        self.profiling_log.append(load_weight_log)

    def profiling_load_cache_stage(self,i , j , k):
        torch.cuda.synchronize()

        load_cache_slot = self.load_cache_start_events[i][j][k].elapsed_time(self.load_cache_ready_events[i][j][k]) * 1e+3
        load_cache_log = {"name": "load_cache", "ph": "X", "pid": 0, "tid": "6. load-cache",
                     "dur": load_cache_slot,"ts": self.get_ts(self.load_cache_start_events[i][j][k]),
                     "args": {"gen-len-id": i,"layers-id": j,"gpu-id": k}, "cname": "cq_build_running"}
        self.profiling_log.append(load_cache_log)

    def profiling_load_hidden_stage(self,i , j , k):
        torch.cuda.synchronize()

        load_hidden_slot = self.load_hidden_start_events[i][j][k].elapsed_time(self.load_hidden_ready_events[i][j][k]) * 1e+3
        load_hidden_log = {"name": "load_hidden", "ph": "X", "pid": 0, "tid": "2. load-hidden",
                     "dur": load_hidden_slot,"ts": self.get_ts(self.load_hidden_start_events[i][j][k]),
                     "args": {"gen-len-id": i,"layers-id": j,"gpu-id": k}, "cname": "rail_response"}

        self.profiling_log.append(load_hidden_log)


    def profiling_compute_stage(self,i , j , k):
        torch.cuda.synchronize()

        compute_slot = self.compute_start_events[i][j][k].elapsed_time(self.compute_ready_events[i][j][k]) * 1e+3
        compute_log = {"name": "compute", "ph": "X", "pid": 0, "tid": "3. compute",
                     "dur": compute_slot,"ts": self.get_ts(self.compute_start_events[i][j][k]),
                     "args": {"gen-len-id": i,"layers-id": j,"gpu-id": k}, "cname": "black"}

        self.profiling_log.append(compute_log)

    def profiling_store_hidden_stage(self, i, j, k):
        torch.cuda.synchronize()

        store_hidden_slot = self.store_hidden_start_events[i][j][k].elapsed_time(
            self.store_hidden_ready_events[i][j][k]) * 1e+3
        store_hidden_log = {"name": "store_hidden", "ph": "X", "pid": 0, "tid": "4. store-hidden",
                           "dur": store_hidden_slot, "ts": self.get_ts(self.store_hidden_start_events[i][j][k]),
                           "args": {"gen-len-id": i, "layers-id": j, "gpu-id": k}, "cname": "rail_response"}

        self.profiling_log.append(store_hidden_log)

    def profiling_store_cache_stage(self,i , j , k):
        torch.cuda.synchronize()

        store_cache_slot = self.store_cache_start_events[i][j][k].elapsed_time(self.store_cache_ready_events[i][j][k]) * 1e+3
        store_cache_log = {"name": "store_cache", "ph": "X", "pid": 0, "tid": "5. store-cache",
                     "dur": store_cache_slot,"ts": self.get_ts(self.store_cache_start_events[i][j][k]),
                     "args": {"gen-len-id": i,"layers-id": j,"gpu-id": k}, "cname": "cq_build_running"}
        self.profiling_log.append(store_cache_log)






    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        # It first constructs the path to the weight file by concatenating the path argument (which is the directory where the weight file is stored), the name of the configuration (config.name), and the string "-np". This path is then expanded using os.path.abspath and os.path.expanduser, which ensure that any relative paths are resolved correctly and that the path includes the user's home directory.
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)
        # Finally, it initializes the weight tensor for the layer by calling the layer's init_weight method and passing in the weight_home tensor for that layer (i.e., the tensor that holds the weights for that layer on the home device) and the expanded path to the weight file.
        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        """
        This function load_weight is responsible for loading the weight of a particular layer for a given token index i, layer index j and GPU batch index k.
        @return: Finally, once the weight loading is complete, the weight is stored in weight_read_buf,
        which is used to compute the output for the current token index.
        """
        # Handle corner cases
        # This block of code checks if the layer index j is equal to the number of layers in the model.
        # If it is, it means that we have completed loading all the weights for the current token index i.
        # Therefore, we move to the next token index by incrementing i by 1 and reset j to 0.
        # If i is equal to the maximum token index execute_gen_len, it means we have finished generating the output sequence,
        # and we return from the function.
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        # This block of code loads the weight of layer j for token index i and GPU batch k from weight_home to weight_read_buf.
        # The overlap argument specifies whether to overlap the weight loading operation with other computations.
        # If overlap is set to True, we use a CUDA stream (load_weight_stream) to overlap the weight loading with other operations.
        # This can help reduce the overall execution time. If overlap is set to False, the weight loading operation is performed synchronously.
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                #print("i"+str(i))

                self.profile_mark_load_weight_start(i,j,k)
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
                self.profile_mark_load_weight_end(i,j,k)
                self.profiling_load_weight_stage(i,j,k)
        else:
            self.profile_mark_load_weight_start(i, j, k)
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
            self.profile_mark_load_weight_end(i, j, k)
            self.profiling_load_weight_stage(i, j, k)

    def delete_weight(self, j, k):
        """
        The delete_weight method of the OPTLM class deletes the weight tensors from GPU memory
         that were used in the previous num_gpu_batches forward or backward passes, starting from the k-th batch
        """
        # The if condition checks if k is equal to 0, which means that the current batch is the first batch of the current sequence.
        # If this is not the case, then there's no need to delete the weight tensors because they are still needed for the next batches.
        if k == 0:
            for x in self.weight_home[j].pop():
                # This if condition checks whether x is an instance of the ValueHolder class. The ValueHolder class is a helper class that holds a reference to a PyTorch tensor and provides methods to access and modify that tensor.
                if isinstance(x, ValueHolder):
                    # If x is a ValueHolder instance, then we retrieve and remove the first tensor from its internal list by calling pop() on x. We're using another for loop to iterate over this internal list of tensors, because x can potentially hold multiple tensors.
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        # 不同layer有不同的init策略
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        # If k is equal to self.num_gpu_batches, then it means we have completed processing all the batches for the current layer.
        # So, we reset k to 0 and move to the next layer (j is incremented).
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        # if j == self.num_layers:
        # - If j is equal to self.num_layers, then it means we have completed processing all the layers.
        # So, we reset j to 0 and move to the next generation (i is incremented).
        if j == self.num_layers:
            j = 0
            i += 1
            # if i == self.execute_gen_len:
            # - If i is equal to self.execute_gen_len, then it means we have completed processing all the generations, and we just return from the function.
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                # self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
                # - We call the load_cache function of the j-th layer to load the cache from cache_home to cache_read_buf.
                # The cache for the j-th layer and k-th batch is loaded for the i-th generation.
                self.profile_mark_load_cache_start(i,j,k)
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
                self.profile_mark_load_cache_end(i,j,k)
                self.profiling_load_cache_stage(i,j,k)
        else:
            self.profile_mark_load_cache_start(i, j, k)
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
            self.profile_mark_load_cache_end(i, j, k)
            self.profiling_load_cache_stage(i, j, k)

    def store_cache(self, i, j, k, overlap=True):
        """
        It is responsible for storing the computed cache values in memory for future use during the forward pass.
        """
        # Handle corner cases
        # The method first handles the corner cases where k or j becomes negative, or i reaches the end of the generation length.
        # If any of these conditions is true, the method returns without storing any cache.
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.profile_mark_store_cache_start(i,j,k)
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
                self.profile_mark_store_cache_end(i,j,k)
                self.profiling_store_cache_stage(i,j,k)
        else:
            self.profile_mark_store_cache_start(i, j, k)
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
            self.profile_mark_store_cache_end(i, j, k)
            self.profiling_store_cache_stage(i, j, k)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        """
        It is responsible for loading hidden states from previous iterations into the model's compute buffer.

        """
        # Handle corner cases
        # The function first handles some corner cases by updating j, k, and i based on their current values. If i has reached execute_gen_len (which is the length of the generation that is being executed), then the function simply returns without doing anything.
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        # the function loads the hidden states from previous iterations into the model's compute buffer.
        # If j is 0, then the hidden state is loaded from either the input ids (if i is 0) or the last generated token (if i is not 0).
        self.profile_mark_load_hidden_start(i, j, k)

        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)


        self.hidden[i][j][k].store(val)
        self.profile_mark_load_hidden_end(i,j,k)
        self.profiling_load_hidden_stage(i,j,k)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        self.profile_mark_store_hidden_start(i,j,k)
        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)
        self.profile_mark_store_hidden_end(i, j, k)
        self.profiling_store_hidden_stage(i,j,k)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.profile_mark_compute_start(i,j,k)
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)
        self.profile_mark_compute_end(i,j,k)
        self.profiling_compute_stage(i,j,k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)
    # 这段代码定义了一个名为update_attention_mask的函数，用于在每个生成步骤中更新模型的注意力掩码。
    # 函数的第一个参数i表示生成的步骤数，第二个参数k表示 GPU 批次的索引。如果步骤数i大于0，则直接扩展已有的掩码，否则将从输入中创建新的掩码。
    # 首先，如果i大于0，则调用device.extend_attention_mask函数来将掩码扩展一个新的位置。
    # 如果i等于0，则需要创建新的掩码，首先计算出 GPU 批次中当前处理的样本在输入中的位置，然后根据这些位置从output_ids中选择相应的片段作为输入。
    # 接下来，根据输入创建一个掩码，将其载入一个ValueHolder对象中并存储到attention_mask数组中，以便后续生成步骤中使用。
    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)
    # 输入参数包括输入文本（可以是np.array或者List[List[int]]）、最大生成文本长度（max_new_tokens）、是否采样（do_sample）、
    # 生成温度（temperature）、停止标记（stop）、调试模式（debug_mode）、生成文本长度截断（cut_gen_len）和详细程度（verbose）。

    # 输出是一个二维数组，其中每个元素都是一个代表标记 ID 的整数。该输出数组的第一维代表不同的输入文本，第二维则是生成文本的长度，
    # 即 prompt_len + gen_len。
    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len

        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids 这段代码主要实现了生成文本时的输出，即 output_ids，以及 stopped 的计算。
        #
        # 首先，代码通过 np.full() 函数创建一个 shape 为 (len(task.inputs), prompt_len + gen_len) 的数组，
        # 其中 len(task.inputs) 表示要生成的文本的数量，prompt_len + gen_len 表示每个文本的长度，
        # 其中 prompt_len 表示输入的 token 数量，而 gen_len 则表示要生成的新的 token 数量。
        # 这个数组的初始值被设置为 self.config.pad_token_id，默认情况下表示 padding token，
        # 也就是在 transformer 模型中用来填充序列长度的标记。

        # 最后，代码创建了一个 shape 为 (len(task.inputs), 1) 的布尔数组 stopped，用于表示每个文本生成是否停止。初值全部设置为 False。
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        # 接下来，代码将输入的 task.inputs 转换成 np.array 格式，并将其赋值给 output_ids 数组中的前 prompt_len 列。
        # 这样就将输入的 token 添加到了 output_ids 中，同时 output_ids 中后面的列保持不变。
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache 这段代码的作用是初始化模型的缓存和隐藏层。具体的作用如下：
        # 根据模型的层数和 GPU 批处理数量，清空模型的缓存区域，
        # 包括 self.cache_home、self.cache_read_buf、self.cache_write_buf 和 self.weight_read_buf。
        # 根据 GPU 批处理数量，清空模型的 attention_mask。
        # 初始化模型的隐藏层，创建大小为 (gen_len, num_layers, num_gpu_batches) 的 3D 数组。
        # 其中 gen_len 是生成的新 tokens 数量，num_layers 是模型的层数，num_gpu_batches 是 GPU 批处理数量。
        # ValueHolder 是一个自定义类，用于存储层级间的激活值。
        # 设置任务并初始化缓存，即调用 self.set_task(task) 和 self.init_cache(j, k) 函数，为每个缓存区域分配内存空间。
        # 如果使用 CPU 进行缓存计算，则调用 self.env.cpu.init_attention_compute_workspace 函数初始化 CPU 的计算空间。
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        # Generate
        if debug_mode is None:
            if not overlap:
                # No overlap, easy to understand, suitable for debugging
                self.generation_loop_normal()
            else:
                # Overlap I/O and compute
                if num_gpu_batches == 1:
                    self.generation_loop_overlap_single_batch()
                else:
                    self.generation_loop_overlap_multi_batch()
        elif debug_mode == "fewer_batch":
            # Run fewer layeres and batches for debugging
            if num_gpu_batches == 1:
                self.generation_loop_debug_single_batch()
            else:
                self.generation_loop_debug_multi_batch()
        elif debug_mode == "breakdown":
            # No overlap, fewer batches, execution time breakdown
            self.generation_loop_debug_normal()
        else:
            raise ValueError("Invalid debug mode: {debug_mode}")

        # Delete cache
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()
        print("overlap")
        print(args.overlap)
        file_name = "/trace_json/flexGen/" + str(args.model)+"_no_dist_overlap_"+str(args.overlap) + '_num_gpu_batches_' + str(args.num_gpu_batches) + '_gpu_batch_size_' + str(args.gpu_batch_size)+'_prompt_len_'+ str(args.prompt_len)+'_gen_len_'+ str(args.gen_len)  + '_percent_' + str(
            args.percent) + '.json'
        dir_name = file_name[0:file_name.rindex('/')]
        print(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_name, 'w') as outputJson:
            json.dump(self.profiling_log, outputJson)

        return self.output_ids




    # 这段函数实现了生成文本的循环过程，其中的变量execute_gen_len是指定生成的文本长度。
    # 具体来说，这个循环会逐步生成文本，每次生成一个 token，直到达到指定的文本长度为止。
    #
    # 在循环的每一步中，首先会更新注意力掩码，这个掩码指示哪些 token 应该被模型考虑到。
    # 然后，对于每一层和每一个 GPU 批次，会加载相应的权重和缓存，然后计算这一层的输出。
    # 计算过程会用到前面生成的 token、前面的层的输出以及相应的权重，具体计算过程由compute_layer函数实现。
    # 最后，会将这一层的输出存储起来，并将缓存也存储起来以供下一次计算使用。循环最后会生成一段指定长度的文本。
    # 没有pipeline parallelism
    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            # the implementation of zig-zag
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def generation_loop_debug_normal(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(self.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)

            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()

                load_weight_timer.start(self.sync)
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k)
                load_weight_timer.stop(self.sync)

                for k in range(self.num_gpu_batches):
                    load_cache_timer.start(self.sync)
                    self.load_cache(i, j, k)
                    load_cache_timer.stop(self.sync)
                    self.load_hidden(i, j, k)
                    compute_layer_timer.start(self.sync)
                    self.compute_layer(i, j, k)
                    compute_layer_timer.stop(self.sync)
                    self.store_hidden(i, j, k)
                    store_cache_timer.start(self.sync)
                    self.store_cache(i, j, k)
                    store_cache_timer.stop(self.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill_total").stop(self.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {self.num_layers}")

        print(f"#batches prefill:  "
              f"{self.num_layers * self.num_gpu_batches}")
        print(f"#batches decoding: "
              f"{(self.task.gen_len - 1) * self.num_layers * self.num_gpu_batches}")
        print(f"load_weight            (per-layer)"
              f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self):
        """
        The generation_loop_overlap_single_batch() method is called by the generate() method if there's overlap and num_gpu_batches is 1.
        This method generates text using a single batch of data and performs I/O and compute operations concurrently.
        """
        # Prologue
        # This is the prologue where weights are loaded into each cache home of the first layer.
        # self.sync() is called to ensure all the devices have finished their computation.
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                # store_cache can be used to updated the cache of the last layer of the former token
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        """
        difference between multi_batch and one_batch:
        1. Attention Mask: In generation_loop_overlap_single_batch(), the attention mask is updated for each token using self.update_attention_mask(i, 0), while in generation_loop_overlap_multi_batch(), the attention mask is updated for each batch k and token i using self.update_attention_mask(i, k).
        2. Cache Tensor: In generation_loop_overlap_single_batch(), the cache for each layer is updated using self.store_cache(i, j-1, 0), which updates the cache for the previous layer, while in generation_loop_overlap_multi_batch(), the cache for each layer and batch is updated independently using self.store_cache(i, j, k-1).
        3. Hidden Layer Tensor: In generation_loop_overlap_single_batch(), the hidden layers for each layer are updated using self.store_hidden(i, j, 0) and self.load_hidden(i, j, 0), while in generation_loop_overlap_multi_batch() the hidden tensor is updated independently for each batch using self.store_hidden(i, j, k-1) and self.load_hidden(i, j, k+1).
        4. Compute Layer: Both functions compute the layer for token i and layer j, but in generation_loop_overlap_multi_batch(), the computation for each batch k is performed independently by using self.compute_layer(i, j, k).
        """
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)

    def generation_loop_debug_single_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            if i == 0: timers("prefill").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                if i > 0: timers("decoding_gpu_batch").start()
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1)
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches: break
            if batch_ct >= execute_num_batches: break
            if i == 0: timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(self.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(self.num_layers * batch_cost)

    def __del__(self):
        self.delete_all_weights()


def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    """
    The run_flexgen function is used to perform benchmarking and generate text using the optimized language model
    """
    print(f"<run_flexgen>: args.model: {args.model}")
    # Tokenizer Initialization: Depending on the argument args.model, the function initializes the tokenizer for the specified model.
    # The padding_side is set to "left", indicating that padding will be added to the left of the input.
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    # roles of pytorch_backend: gpu and cpu will be then set to be attributes of env,
    # which will be passed to OPTLM, and then the function in pytorch_backend will be called in different layers in OPTLM
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

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
    # 计算 cache 大小和 隐藏状态大小
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")


    model = OptLM(opt_config, env, args.path, policy,args.gen_len, args.cut_gen_len)
    try:
        print("warmup - generate")
        # Warm-up: Before running the benchmark, the generate method is called with warmup_inputs and max_new_tokens=1. This generates one token of output and warms up the model,
        # potentially filling any caches and reducing the overhead for subsequent generate operations.
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        #  The generate method is invoked again with inputs and max_new_tokens=args.gen_len to generate the actual text.
        #  The debug_mode and cut_gen_len parameters
        #  parser.add_argument("--cut-gen-len", type=int, help="Cut generation length for fast debugging.")
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
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
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    # todo default params
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
