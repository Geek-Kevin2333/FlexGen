"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()

    @property
    def bytes(self):
        """
        The bytes function is a method of the class TorchTensor that calculates the number of bytes required
        to store the tensor data in memory.

        The np.prod(self.shape) part calculates the total number of elements in the tensor by taking the product of all its dimensions.
        For example, for a tensor of shape (2, 3, 4), the total number of elements would be 2 * 3 * 4 = 24.

        The torch_dtype_to_num_bytes[self.dtype] part maps the tensor data type to the number of bytes required to store one element of that type.
        This is necessary because different data types require different amounts of memory.
        For example, a float32 tensor requires 4 bytes for each element, while a float64 tensor requires 8 bytes.
        @return:
        """
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        """
        This is a class method of the TorchTensor class that generates the next name to be used for a tensor.
        It returns a string in the format "t_{count}", where count is an integer representing the number of tensors
        that have been created so far.
        @return:  nextName-->next name to be used for a tensor.
        """
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        """
        This is a class method of the TorchTensor class that creates a TorchTensor object from a torch.
        Tensor object. It takes three arguments:
        @param data: The torch.Tensor object to be wrapped.
        @param device: A TorchDevice object representing the device on which the tensor is stored.
        @param name: An optional string representing the name of the tensor.
        @return:
        """
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None

    def load_from_np(self, np_array):
        """
        This is a method of the TorchTensor class that loads data from a NumPy array into the tensor. It takes one argument:
        @param np_array: The NumPy array containing the data to be loaded.
        The function checks the device type of the tensor, and loads the data differently depending on the device type.
        If the device type is DISK, the data is written to a file.
        If the device type is COMPRESSED, the data is first converted to a torch.Tensor, compressed,
        and then loaded into the tensor. Otherwise, the data is simply copied into the tensor using self.data.copy_()
        """
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                # 在cpu中压缩
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        """
        This is a method of the TorchTensor class that loads data from a NumPy file into the tensor. It takes one argument:
        The function checks the device type of the tensor, and loads the data differently depending on the device type.
        If the device type is DISK, the data is copied from the file to the tensor.
        Otherwise, the data is first loaded into a NumPy array using np.load(),
        and then loaded into the tensor using self.load_from_np().
        @param filename: The filename of the NumPy file containing the data to be loaded.
        """
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        """
        This is a method of the TorchTensor class that creates a copy of the tensor on another device.
        @param dst: A TorchDevice object representing the destination device.
        @param src_indices: An optional list of slices representing the indices of the tensor to be copied.
        @todo not so familiar with the data transformation operation in pytorch.
        @return:
        """
        if src_indices:
            # If src_indices is specified, the method calculates the shape of the tensor that is to be copied by taking the difference of each slice's stop and start attributes,
            # and concatenating this with the remaining shape of the tensor object.
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape
        # allocate is used to init space
        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices)
        return ret

    def smart_copy(self, dst, src_indices=None):
        """
        The function checks whether the tensor is already on the destination device, and if so,
        returns the tensor and a flag indicating that the tensor was not copied. Otherwise, the function creates a copy of the tensor on the destination device using self.copy(),
        and returns the copied tensor and a flag indicating that the tensor was copied.
        """
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True

    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")


class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU.
    tensor APIs include:allocate, init_attention_compute_workspace, attention_mask相关, init_cache_one_gpu_batch
    computation APIs include:opt_input_embed, opt_output_embed, mha, mha_gen, mlp
    """

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link
    # 首先，代码检查设备类型，如果是CPU，它将pin_memory设置为True，否则设置为False。
    # 然后，根据传递的dtype，将它从Numpy数据类型转换为PyTorch数据类型。接下来，代码使用PyTorch的empty函数在该设备上创建一个新的张量，该张量的形状和数据类型由shape和dtype指定。
    # 如果pin_memory设置为True，则将其设置为该张量的pin_memory参数。最后，该函数返回一个TorchTensor对象，通过调用create_from_torch函数来从PyTorch张量中创建。
    # 这个对象是对PyTorch张量的包装，它包括该张量所在的设备和一些额外的元数据，如名称等。
    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass
    # 这段代码是用来初始化注意力机制计算过程中的工作空间的，其目的是为了提高计算效率。具体来说，如果设备类型不是 CPU，那么该函数就直接返回；
    # 否则，如果 policy 的压缩缓存选项为 False，那么就会分配计算空间。这里的计算空间包括了 k_cache 和 v_cache 两个变量，
    # 它们分别表示缓存 Q, K, V 的张量，用来在计算中重复使用而不需要频繁地从内存中读取。
    # 这里的 shape 参数包括了最大的序列长度 max_seq_len，batch_size 的数量 b，头数 n_head 以及每个头部的维度 head_dim。
    # 如果 policy 的压缩缓存选项为 True，那么就会调用另外一个函数 compressed_device.init_attention_compute_workspace() 来进行初始化工作
    # The purpose of this function is to reduce the I/O operations of Q, K, and V tensors during attention calculation.
    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # todo why？Only CPU requires this fp32 workspace
        # If the compress_cache option of the policy object is False, then the function allocates the workspace needed for the attention computation.
        # Specifically, the function initializes two workspace caches, k_cache and v_cache, for the Q, K, and V tensors used by the attention computation.
        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            # A loop is used to iterate through the layers and initialize the workspace caches for each layer.
            # If the sep_layer option is true (layers are separated into self-attention and MLP layers), then there will only be one cache for both MX and self-attention.
            # Otherwise, there will be two caches, one for self-attention and one for MX layers.

            for i in range(1 if policy.sep_layer else 2):
                # The shape of each of the caches is set as (max_seq_len, b * n_head, head_dim) which corresponds to the shape of Q, K, and V tensors used in attention computation.
                shape = (max_seq_len, b * n_head, head_dim)
                # allocate can help to speed up todo but why?
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        # It first applies a boolean condition, checking if any element in the token_ids tensor is not equal to the pad_token_id value.
        # This creates a boolean tensor with the same shape as token_ids, with each element indicating whether that element in token_ids is a pad token or not.
        data = token_ids.data.ne(pad_token_id)
        # The next line checks whether the first element of the donate parameter is True.
        # If it is, then it deletes the token_ids tensor.
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        # bs --> batchSize
        # It first gets the batch size of the attention_mask tensor.
        # The next line constructs a new tensor by concatenating the attention_mask data tensor with a tensor of ones with shape (bs, 1) along the second (column) dimension.
        # This is done using the PyTorch concat method.

        # Why we need this function
        # The extend_attention_mask function is used to add an extra column of ones to the attention mask tensor.
        # The attention mask is a binary mask indicating which positions in the input sequence should be attended to and which should be ignored.
        # The extra column of ones ensures that every position in the input sequence is attended to at least once,
        # even if the original attention mask has all zeros for that position.

        # This is especially useful when working with masked language modeling tasks such as BERT.
        # In these tasks, a special token (usually [MASK]) is inserted in place of some tokens in the input sequence,
        # and the model is trained to predict the original token based on the context.
        # However, in order to ensure that the model attends to the [MASK] token during training,
        # the attention mask needs to be modified to include the [MASK] token in the attended positions.

        # By adding an extra column of ones to the attention mask tensor, the model is able to attend to all positions in the input sequence,
        # including the [MASK] token, even if the original attention mask does not include it.

        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)
    # 这两个函数是用于自然语言处理中的transformer模型中的embedding层的输入和输出的优化的。
    # 在第一个函数 opt_input_embed 中，给定输入tokens和attention_mask，以及需要使用的embedding weights和position weights，
    # 然后通过调用PyTorch的F.embedding函数获得token embedding和positional embedding，并将它们相加得到最终的embedding表示。
    # 最后将其返回为一个 TorchTensor 对象。
    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        """
        @param inputs: the tokenized input sequence of shape (batch_size, sequence_length).
        @param attention_mask: a binary mask tensor of shape (batch_size, sequence_length) indicating the valid positions of the input sequence. If a token is masked, its corresponding vector will be filled with 0's.
        @param w_token: a tensor of shape (vocab_size, embedding_dim) containing the token embeddings for all tokens in the vocabulary.
        @param w_pos: a tensor of shape (max_sequence_length, embedding_dim) containing the positional embeddings.
        @param pad_token_id: an integer that represents the id of pad token which is added at the end of the sequences to make them the same length.
        @param donate: a list containing two boolean values, specifying whether to free the memory occupied by inputs and attention_mask.
        @return: TorchTensor opt_input_embed() returns an embedding tensor of shape (batch_size, sequence_length, embedding_dim) consisting of the token embeddings and positional embeddings corresponding to the input sequence with valid positions determined by the attention_mask.
        """
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        # The function then computes the token embeddings, which are a lookup into the w_token tensor using the F.embedding function.
        # The F.embedding function is used to generate embeddings for the sequences.
        # pad_token_id is used as a padding token, which represents that the token has no useful meaning for the model training or inference.
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)

        # pos embedding
        # Next, the positional embeddings are calculated.
        # This is done using the positions tensor; the positions tensor is the cumulative sum of the attention_mask matrix along the sequence length dimension.
        # Positions that are masked are discarded, meaning they are set to 0.
        # By doing this we set the positions of the masked tokens to 0, and the positions of the un-masked tokens to their respective position in the sequence.
        # This creates positions for all the un-masked tokens.
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)
    # 第二个函数 opt_output_embed 是用于将transformer模型的hidden state转化为输出token的logits，
    # 它接受一个hidden state输入以及需要使用的layer normalization weights、bias weights和输出embedding weights。
    # 它先对hidden state进行layer normalization，然后通过调用PyTorch的F.linear函数，将hidden state映射到输出embedding space，
    # 然后将最后一个时间步的logits取出来，并根据是否需要进行采样和temperature的设置，返回一个被采样的token id或argmax之后的token id，
    # 最后将其返回为一个 TorchTensor 对象。
    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        """
        @param inputs: a tensor of shape (b, s, h), representing the input embeddings.
        @param w_ln: a tensor of shape (h,), representing the weights of the layer normalization.
        @param b_ln: a tensor of shape (h,), representing the bias of the layer normalization.
        @param w_token: a tensor of shape (h, vocab_size), representing the weights of the output embedding layer.
        @param donate: a list containing a single boolean value. This variable is ignored if its first element is False. Otherwise, if it's True, the inputs tensor is deleted using inputs.delete(), which removes the tensor from the computation graph and free the memory.
        @param do_sample: a boolean variable indicating whether to sample from the probabilities (True) or take the most probable sequence (False).
        @param temperature:  a temperature value used in the softmax function if the do_sample variable is True.
        @return: TorchTensor
        """
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
        # the shape of the inputs tensor is unpacked to three variables:
        # b for the batch size, s for the sequence length, and h for the hidden size.
        b, s, h = inputs.shape
        # the layer normalization is applied to the inputs tensor using the F.layer_norm function.
        # The weight argument is set to w_ln.data, and the bias argument is set to b_ln.data.
        # As a result, the hidden tensor is a layer-normalized version of the inputs tensor.
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        # the output embedding is computed.
        # First, a linear transformation of the hidden tensor with the weights w_token is computed using the F.linear function,
        # and the result is stored in the logits variable.
        # Then, the last_token_logits tensor is obtained by selecting the last token(last sequence) from the logits tensor.
        logits = F.linear(hidden, w_token.data)
        last_token_logits = logits[:,-1,:]
        # if do_sample is True, the function applies the softmax function with temperature temperature to the last_token_logits tensor to obtain a tensor probs of probabilities (of shape (b, vocab_size)).
        # The tensor ids of shape (b, 1) is obtained by performing multinomial sampling over probs once, using the torch.multinomial function with argument num_samples=1.
        # If do_sample is False, then the most probable index in last_token_logits tensor is obtained using the argmax method with argument dim=1, and the resulting ids tensor has shape (b, 1).
        if do_sample and not temperature < 1e-5:
            # The line probs = torch.softmax(last_token_logits / temperature, dim=-1) uses the popular softmax function to create a probability distribution over the elements of last_token_logits.
            # The softmax function normalizes the values in last_token_logits by exponentiating them and dividing them by their sum along the last dimension of last_token_logits.
            # The parameter temperature scales the values before normalization, thus multiplying the logits by a small value (high temperature) makes them closer to equal allowing the probabilities to form a more uniform distribution, and multiplying them by a large value (low temperature) narrows the distribution.
            # The torch.softmax function used here takes the tensor of values, the dim parameter specifies the last dimension over which the softmax should be applied.
            # In this case, it is the last dimension of last_token_logits tensor, which represents the vocabulary size, and it is set to -1.
            #
            # The resulting tensor probs has the same shape as last_token_logits, and each element probs[i, j] represents the probability of the j-th token in the vocabulary to follow the i-th token in the input sequence, given the output of the neural network.
            #
            # The next line, ids = torch.multinomial(probs, num_samples=1), generates a tensor ids of integers, by sampling from the probability distribution defined by probs.
            # The torch.multinomial function randomly samples num_samples=1 integers based on the probabilities in the probs tensor. The resulting ids tensor has shape (batch_size, 1) containing the sampled integer indices.
            #
            # Overall, combining these two lines of code allows us to sample a sequence of tokens based on the probabilities produced by the model from the given input sequence.
            # Sampling from a probability distribution rather than taking the argmax allows the model to generate more diverse and interesting output during inference.
            # By using the temperature parameter to adjust the strength of the softmax, the sampling strategy can be more or less conservative which further influences the output generated by the model.
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        Overall, init_cache_one_gpu_batch() function is crucial because it allocates memory for the caching mechanism used in the Transformer model.
        While carrying out inference, this memory helps the model to avoid recomputing computations which were already performed on certain portions of the input sequence.
        @param config: an instance of a TransformerConfig class that sets up the hyperparameters for the Transformer model.
        @param task: an instance of a GPTTask class that provides the context of the task, such as the prompt length, the generation length, etc.
        @param policy: an instance of a StreamingDataParallelPolicy class that optimizes the parallelism strategy for the Transformer model.
        @return: k_cache, v_cache
        """
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        # The shape of the cache buffers, a tuple of three integers, is calculated based on hyperparameters.
        # The first dimension of the shape is prompt_len + gen_len - 1, which represents the number of total positions in the sequence where the key-value pairs are to be cached.
        # The second dimension of the shape is gpu_batch_size * num_head, which represents the number of parallel streams to cache, and is equal to the batch size multiplied by the number of attention heads.
        # The third dimension of the shape is hidden_size // num_head, which represents the size of the value/key vectors in each attention head. Given that the hidden_size is evenly divisible by the number of attention heads num_head, hidden_size // num_head represents the size of the value/key vectors per head.

        # The shape is a tuple of three integers,
        # where the first dimension represents the number of total positions in the sequence where the key-value pairs are to be cached.
        # This is computed as prompt_len + gen_len - 1 since each generated token depends on the previous tokens as well as the prompt.
        # The second dimension represents the number of parallel streams to cache, which is equal to the batch size multiplied by the number of attention heads (gpu_batch_size * num_head).
        # The third dimension represents the size of the value/key vectors in each attention head, which is hidden_size // num_head.
        #
        # By setting the cache shape in this way, the model can store and retrieve key-value pairs for each position in the sequence and for each attention head,
        # without needing to recompute them in subsequent layers during inference.
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        # Note that the pin_memory flag is false,
        # and can be set to true if the GPU has limited memory and needs to access the data from another allocator.
        # If pin memory is enabled, memory is allocated only on the host and not on the GPU,
        # and the memory is then streamed to the GPU.
        # However, this means additional overhead.
        # Thus, when pin memory is disabled, memory is allocated directly on the GPU, reducing host to GPU transfer overhead.
        pin_memory = False
        # memory buffers (ndarray) that store the cached key/value pairs, respectively.
        # Finally, the function returns k_cache and v_cache which are the memory buffers to store the cached key-value pairs
        # used during inference for the given task and configuration.
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache

    # 这是一个实现了多头自注意力机制的函数。主要输入参数包括inputs(输入向量序列)，attention_mask(遮蔽向量序列)，
    # w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln(参数矩阵和偏移量)，n_head(头数)，donate(是否释放内存)，
    # compress_cache和comp_config(压缩缓存参数)。函数首先将压缩过的参数解压缩，然后将输入向量序列进行归一化。
    # 接着将归一化后的输入向量通过参数矩阵和偏移量分别生成查询矩阵q，键矩阵k和值矩阵v。
    # 这些矩阵分别经过变换后，将注意力权重与遮蔽向量相乘并进行softmax计算，最后根据计算出的权重对值矩阵v进行加权求和得到最终输出值。
    # 最后，函数还将生成的矩阵进行压缩，并根据参数donate释放内存。函数返回输出值以及经过压缩的键矩阵k和值矩阵v
    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase).
        """
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        # (batch_size,sequence_length,hidden_size)
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5
        # Then the input vector sequence is normalized using layer norm.
        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        # After transformation of these matrices, the attention weight is found by taking the dot product of the query matrix q and key matrix k.
        # Softmax operation is performed on the generated attention weight.
        # The value matrix v is then weighted and summed according to the calculated weight to obtain the final output value.
        # batch matrix multiplication, which expects the last two dimensions of the matrices to match.
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        # idx is a tensor of indices from 0 to s-1, where s is the length of the input sequence. It is created using torch.arange,
        # which returns a tensor with evenly spaced values between a given start and end value.
        idx = torch.arange(s, device=self.dev)
        # causal_mask is a binary mask tensor of shape (1, 1, s, s)
        # that indicates whether a token at position i can attend to tokens at positions greater than i.
        # It is created by comparing the idx tensor with its transpose using <= operator and reshaping the result to the desired (1, 1, s, s) shape.
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        # mask is a binary mask tensor of shape (batch_size, 1, 1, s) that combines the attention mask tensor with the causal mask tensor.
        # It is created by broadcasting the attention mask tensor to the shape (batch_size, 1, 1, s) and taking an element-wise logical AND (&) operation with the causal mask tensor.
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        # After computing the dot product between the query matrix q and key matrix k, the resulting attention weight has a shape of (batch_size * n_head, seq_len, seq_len).
        # These attention weights are then reshaped into a 4-dimensional tensor of shape (batch_size, n_head, seq_len, seq_len) using view.
        # The first dimension groups the attention weights by batch, the second dimension by head,
        # and the third and fourth dimension represent the source and target positions, respectively.
        attn_weights = attn_weights.view(b, n_head, s, s)
        # The attention mask created earlier (with shape (batch_size, 1, 1, seq_len)) is then broadcasted to the same shape as the attention weights
        # and combined with the causal mask.
        # This creates a tensor of shape (batch_size, n_head, seq_len, seq_len) where the attention weights corresponding to masked positions are replaced with a large negative value (-1e4 in this case).
        # This ensures that the softmax operation used to normalize the attention weights assigns a near-zero probability to these masked positions.
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        # The masked attention weights are then reshaped again into a 3-dimensional tensor of shape (batch_size * n_head, seq_len, seq_len)
        # and the softmax operation is applied along the last dimension.
        # This generates a tensor of shape (batch_size * n_head, seq_len, seq_len) containing the attention weight for each token with respect to every other token in the input sequence.
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        # The attention weight tensor is then used to weight the value matrix v using a batch matrix multiplication with torch.bmm. The resulting tensor of shape (batch_size, n_head, seq_len, head_dim) is transposed
        # and reshaped to (batch_size, seq_len, head_dim * n_head) to concatenate the outputs of the different heads.
        value = value.transpose(1, 2).reshape(b, s, h)
        # Finally, a linear transformation is applied to the concatenated values with the weight matrix w_out and bias b_out to generate the final output tensor.
        value = F.linear(value, w_out.data, bias=b_out.data)
        # The output tensor is added to the input tensor inputs with an in-place addition using add_ and the updated tensor is returned.
        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        #
        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v

    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase).
        The function also takes in pre-trained weight and bias matrices for calculating query (w_q, b_q), key (w_k, b_k), value (w_v, b_v), output (w_out, b_out), and layer normalization (w_ln, b_ln).
        The function also takes in a cached key (k_cache) and value (v_cache) matrices for speeding up attention computation during subsequent time steps.
        differences between mha and mha_gen:
        1.Input: mha takes in a single input vector per call, while mha_gen takes in the entire sequence of input vectors at once.
        2.Purpose: mha is used during the encoding phase of the Transformer to compute attention for each input vector separately, while mha_gen is used during the decoding phase to compute attention for the entire sequence at once.
        3.Caching: mha does not use caching, while mha_gen uses cached key and value matrices for the entire sequence of input vectors to speed up attention calculation.
        4.Sparsity: mha_gen need to consider sparsity
        reasons:These differences are required to ensure efficient computation during the encoding and decoding phases of the Transformer model. During encoding, each input vector is processed separately, thus there is no need for caching or normalization layers. On the other hand, during decoding, the attention must be computed for the entire sequence at each time step, so caching the key and value matrices can significantly speed up the computation. Additionally, normalization layers can help stabilize the training by ensuring that the input distributions have zero mean and unit variance, thus enhancing the model's generalization capability.
        Furthermore, whilst mha expects a sequence of vectors of arbitrary length, mha_gen has to be provided the length of the input sequence in advance, and so, its implementation is modified accordingly.

        """
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        # Both functions compute multi-head self-attention by transforming input vectors inputs into query, key, and value matrices.
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            # If sparse attention is used (based on the attn_sparsity parameter),
            # the function computes the sparse attention value using the _sparse_attention_value method.
            # Otherwise, it uses the _attention_value method to compute the attention value.
            if attn_sparsity >= 1.0:  # Dense attention
                # k_cache.data[:src_s] and v_cache.data[:src_s] get the slice of cached data that has been added till this point, i.e, up to position src_s - 1 for each src_s.
                # This is done to ensure that the newly added key and value matrices are not included, as they were added later in the previous if block.
                # k[src_s - 1:src_s] = k_new and v[src_s - 1:src_s] = v_new then adds the newly computed key and value matrices (k_new and v_new) to the end of the cached key and value matrices (k_cache and v_cache).

                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        # The function returns the final output tensor along with the updated cached key (k_new) and value (v_new) matrices.
        # todo : why shape: (b, 1, h) is useful. what's the effect of value here?
        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s) tgt_s = 1?
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        """
        @param q:The query tensor of shape (batch_size * n_head, tgt_s, head_dim) that represents the current query, i.e., the target sequence positions.
        @param k:The key tensor of shape (batch_size * n_head, head_dim, src_s) that contains the cached key matrices from the entire input sequence.
        @param v:The value tensor of shape (batch_size * n_head, src_s, head_dim) that contains the cached value matrices from the entire input sequence.
        @param mask:The attention mask that signals which positions in the input sequence should be attended to for each target position.
        @param b:
        @param src_s:
        @param tgt_s: In the context of mha_gen function, tgt_s refers to the length of the target sequence, which is the number of time steps for which we want to generate the output.
        During the decoding phase of the Transformer model, the model generates output tokens iteratively, one at a time, for each time step.
        The length of the output sequence is not known beforehand and is generated dynamically during decoding.
        Therefore, during the decoding phase, tgt_s is not a constant and changes after each decoding step. It denotes the current position in the target sequence.
        @param n_head:
        @param head_dim:The dimension of each attention head.
        @return:
        """
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)
    # 该函数实现了稀疏注意力机制。稀疏注意力是一种注意力机制的变种，其中仅考虑与查询向量最相关的一小部分键值对，而不是对所有的键值对进行加权求和。
    # 这可以减少计算复杂度，特别是对于非常长的序列。
    # 函数的输入包括查询向量q，键向量k，新值向量v_new和缓存值向量v_cache，掩码mask，批量大小b，源序列长度src_s，
    # 目标序列长度tgt_s，头数n_head，每个头的维度head_dim，以及稀疏度attn_sparsity。输出为稀疏注意力的值向量。
    # 该函数首先使用_attention_weights函数计算注意力权重矩阵，然后根据稀疏度将最相关的一小部分键值对挑选出来。
    # 然后，它将这些键值对从缓存向量中复制到一个临时的v_buf向量中。最后，使用注意力权重矩阵和值向量v计算注意力的值向量，并将其重塑为目标形状。
    # 请注意，此函数的实现使用了一些CUDA特定的代码路径，因此它需要在CUDA环境中运行才能正常工作。
    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        """
        _sparse_attention_value() method is a specialized version of attention used in mha_gen to compute the attention scores for sparsely sampled positions of the input sequence for the current position in the output sequence.
        @param q: The query tensor of shape (batch_size * n_head, tgt_s, head_dim) representing the current query, i.e., the target positions.
        @param k:
        @param v_new: The tensor of shape (b * n_head, head_dim) containing the newly computed value matrix for the current time step.
        @param v_cache: The cache of value matrices for the entire input sequence, represented as a tuple (v_home, v_buf) which contains tensors of shape (src_s, b*n_head, head_dim) and (topk+1, b*n_head, head_dim) respectively.
        @param mask:
        @param b:
        @param src_s:
        @param tgt_s:The current position in the target sequence.
        @param n_head:
        @param head_dim:
        @param attn_sparsity:
        @return:
        """
        # shape: (b * n_head, 1, s)
        #  First, the attention weights are calculated using _attention_weights() method. It is similar to the one used in the non-sparse attention. However, the attention weights are now restricted to only the top k+1 positions instead of the entire input sequence. The top k positions are selected and used for the attention calculation,
        #  and the final position is used to represent all the positions not selected.
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # topk is the number of positions to attend to with the highest attention weights.
        # The fraction attn_sparsity is multiplied by the number of possible positions of the input sequence except the last one since it is used to represent the remaining unselected positions.
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        # topk_weights and topk_indices are obtained by selecting the topk positions with the highest attention scores, from the computedattn_weights`.
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(b * n_head, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        # attn_weights is reconstructed with the top k positions with the highest attention weights concatenated with the attention weight for the final segment.
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, b*n_head, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        # The buffer tensor v_buf is updated with values from the updated v_cache tensor using PyCUDA's general_copy() method.
        # The topk_indices represent the index of the selected positions for current output position, which are stored in v_buf using general_copy().
        # The method then synchronizes the device to ensure that the operation completes before the next kernel launch.
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        # v tensor is first loaded with the required segments of v_buf, including the newly computed value matrix in the last position of v_buf followed by reshaping so that it matches the attention weight tensor shape.
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        """
        This is a function in PyTorch that implements a mixed-device attention mechanism.
        The function computes attention on both the GPU and CPU devices. The key and value tensors for the previous tokens are stored in cache on both the GPU and CPU,
        so the function first splits these tensors into separate variables for the GPU and CPU.
        The function then performs attention on the GPU using the key and value tensors for the previous tokens stored on the GPU,
        and on the CPU using the key and value tensors for the previous tokens stored on the CPU.
        @param q:
        @param k_cache: a tuple of cached key tensors for the previous tokens, stored on both GPU and CPU
        @param v_cache: a tuple of cached value tensors for the previous tokens, stored on both GPU and CPU
        @param k_new: key tensor for the current tokens
        @param v_new: value tensor for the current tokens
        @param mask:
        @param b:
        @param src_s: source sequence length
        @param tgt_s: target sequence length
        @param n_head:
        @param head_dim:
        @return:
        """
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        #  The GPU part is computed first by extracting the query tensor for the current tokens,
        #  the key tensor for the previous tokens on the GPU, and the value tensor for the previous tokens on the GPU.
        #  These tensors are then reshaped and permuted to the appropriate dimensions for the attention computation,
        #  and the attention mask is applied. The attention value is then computed using the _attention_value function.
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        # The CPU part is computed similarly, but using the key and value tensors for the previous tokens stored on the CPU instead.
        # The query tensor for the current tokens is first extracted and converted to CPU float.
        # The key and value tensors for the previous tokens on the CPU are then reshaped and permuted,
        # and the attention mask is applied. The attention value is computed using the _attention_value function.
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)
        # The final attention value is obtained by concatenating the GPU and CPU attention values along the batch dimension,
        # and returning the concatenated tensor.
        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        """
        This is a function in PyTorch that implements a multilayer perceptron (MLP) with layer normalization.
        The MLP consists of two linear layers with a ReLU activation function in between.
        @param inputs:
        @param wi: the weight tensor for the first linear layer
        @param bi: the bias tensor for the first linear layer
        @param wo: the weight tensor for the second linear layer
        @param bo: the bias tensor for the second linear layer
        @param w_ln: the weight tensor for the layer normalization layer
        @param b_ln: the bias tensor for the layer normalization layer
        @param donate: a boolean flag indicating whether to free the memory of the input tensor after the operation
        @return:
        """
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache

    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice
SEG_DIM = 1

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""
    # 初始化TorchMixedDevice实例。base_devices是一个列表，包含所有支持TorchTensor实例的物理设备。
    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices
    # 在多个物理设备上为张量分配内存。shape是张量形状，dtype是数据类型，seg_lengths是一个表示每个设备存储的张量分段长度的列表。
    # pin_memory是一个可选的布尔值，表示是否将张量固定在主机内存中，以加速数据传输。
    # the seg_lengths, which is a list of segment lengths to be allocated across devices
    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None):
        # The function first checks that the sum of the segment lengths is equal to the size of the tensor along the SEG_DIM dimension.
        # This dimension is used to partition the tensor across multiple devices.
        assert sum(seg_lengths) == shape[SEG_DIM]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        # Then, the function initializes seg_points as a list of segment start positions,
        # which is used to calculate the segment lengths for each device.
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        # Next, the function iterates over the available devices, and for each device, it calculates the segment length and shape,
        # and then allocates the memory for the corresponding tensor segment using the allocate method of the device.
        # If the segment length is zero, the function appends None to the tensors list.
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:SEG_DIM] + (seg_len,) + shape[SEG_DIM+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))
        # Finally, the function returns a TorchTensor object that wraps the tensor segments allocated on multiple devices, along with the segment start positions and other metadata.
        # This tensor can then be used for computation on multiple devices using PyTorch's distributed data parallelism.
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (tensors, seg_points), self, name=name)

    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()
    # 在一个GPU批次上初始化缓存，返回键值缓存的TorchTensor实例。
    def init_cache_one_gpu_batch(self, config, task, policy):
        """
        @param config: an instance of a TransformerConfig class that sets up the hyperparameters for the Transformer model.
        @param task: an instance of a GPTTask class that provides the context of the task, such as the prompt length, the generation length, etc.
        @param policy: an instance of a StreamingDataParallelPolicy class that optimizes the parallelism strategy for the Transformer model.
        @return:
        """
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        # The second dimension represents the number of parallel streams to cache,
        # which is equal to the batch size multiplied by the number of attention heads (gpu_batch_size * num_head).
        # The third dimension represents the size of the value/key vectors in each attention head, which is hidden_size // num_head.
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        # Next, the function determines the size of the cache that should be allocated to the GPU, CPU, and disk based on the policy object.
        # If policy.cache_disk_percent is zero, then all the cache will be allocated on the GPU.
        # Otherwise, the cache will be divided among the GPU, CPU, and disk.
        # The sizes are calculated based on the percentages specified in the policy object.
        if policy.cache_disk_percent == 0:
            #  The size of the GPU segment is rounded down to the nearest multiple of num_head, which is the number of attention heads in the transformer model.
            #  This is because the shape of the cache tensor must be divisible by num_head.
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            # The size of the CPU segment is also rounded down to the nearest multiple of num_head.
            len_cpu = shape[SEG_DIM]  - len_gpu
            # is the size of the disk segment. It is calculated by subtracting len_gpu and len_cpu from the total cache size.
            # This segment is used for storing parts of the cache that are not currently needed in memory, to free up space for other parts of the cache.
            len_disk = 0
        else:
            len_gpu = int(shape[SEG_DIM] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[SEG_DIM] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[SEG_DIM] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache


class TorchLink:
    """An I/O link between two devices.
    The TorchLink class defines an I/O link between two devices a and b.
    The link has two bandwidth attributes a_to_b_bandwidth and b_to_a_bandwidth, which represent the transfer speed from device a to device b and from device b to device a, respectively.
    """

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        """
        The io_time method of the TorchLink class calculates the time required to transfer data of size size from a source device src to a destination device dst through the link.
        If the source device is a, the bandwidth attribute a_to_b_bandwidth is used to calculate the transfer time.
        Similarly, if the source device is b, the bandwidth attribute b_to_a_bandwidth is used to calculate the transfer time.
        """
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth

# The function takes in four arguments: dst (the destination tensor),
# dst_indices (a tuple of slice objects representing the indices of the destination tensor that should be copied),
# src (the source tensor), and src_indices (a tuple of slice objects representing the indices of the source tensor that should be copied).
def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()

    The function first checks if dst tensor resides on Mixed or Disk devices, or src tensor resides on Mixed devices, or if either src or dst tensor is compressed.
    In each of these cases, a recursive call to this same function with appropriate arguments is made, effectively enabling copying data between tensors residing on different devices or compression of source and destination tensors before copying them.
    If no special condition is satisfied, src_indices and dst_indices are first set to the full src and dst tensor indices if they were not specified in the arguments.
    Finally, depending on the devices which the tensors reside, the function applies memory-efficient copying approaches, either using specialized CUDA API functions,
    or default PyTorch implementation for normal tensors residing on a single device (not compressed). If both src and dst tensors are on CPU and neither is pinned,
    this implies that they might need to be pinned in memory before copying, and this is done by the pin_memory() function.
    """

    if dst.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        # The first if block recurses over all the sub-devices and sub-slices of a mixed device tensor in the destination.

        # The function throws an error if the source tensor src is itself a mixed device tensor, since recursive calls to general_copy() would be required.
        assert src.device.device_type != DeviceType.MIXED
        # For dst tensor, the function first accesses a list of "segment points" stored in the data attribute of the tensor object.
        # These segment points record the indices that separate the tensor data across its various sub-devices.
        seg_points = dst.data[1]

        # then sets up a for loop that iterates over the list of base_devices of the dst tensor.
        # Each iteration of the loop handles one sub-device of the mixed device dst tensor.
        for i in range(len(dst.device.base_devices)):
            # The loop checks if the sub-device is empty by comparing the segment points of the current device and the next
            if seg_points[i] == seg_points[i+1]:
                continue
            # the code uses Python's built-in slice() function to create empty slice objects,
            # if dst_indices and src_indices are not specified. These will select the entire source and destination tensors.
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            # The function then calculates the new indices for the source and destination tensors based on the indices of the sub-device being processed,
            # by calling the helper function cut_indices().
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            # Finally, the function recursively calls general_copy() with the newly calculated indices and the current sub-device of the mixed device dst tensor,
            # copying data to and from corresponding regions of the src tensor.
            general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices)
    # The second if block similarly handles mixed device source tensor for a copy operation.
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1],
                base=seg_points[i])
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1])
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices)
    # The third if block handles compressed tensors and dispatches to general_copy_compressed function.
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        # In PyTorch, memory pinning is the act of manually specifying that memory regions for tensors should be allocated in such a way that they can be directly accessed by the GPU or other hardware accelerators.
        # When a tensor is pinned, its memory is aligned to the device's memory boundaries, which helps to speed up copies between CPU and GPU by facilitating asynchronous transfers.
        # In general, memory pinning can improve the performance of data transfers between CPU and GPU by reducing overhead and allowing asynchronous transfer.
        # By pinning the memory, the copy operation can be performed asynchronously, which allows the CPU to perform computations while the transfer is being completed.
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0):
    # assert all(x.step is None for x in indices) checks if all slices are simple (that is each slice specifies only start, stop, and step) slices.
    assert all(x.step is None for x in indices)
    # The variable seg is a slice object representing the segment of the tensor being processed
    seg = indices[SEG_DIM]
    # returns a tuple populated with new slices without the segment slice
    return (indices[:SEG_DIM] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[SEG_DIM + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """
    The copy worker thread.
    The copy_worker_func function is a worker function that runs in a separate thread and is responsible for copying data from one device to another.
    The function takes in two arguments:
    queue, which is a queue.Queue object that contains items to be processed,
    and cuda_id, which is the ID of the CUDA device on which the function will execute.

    By using a separate stream for copy operations, we can overlap memory copies with compute operations, which can lead to performance improvements.
    Specifically, the copy operations can be executed asynchronously with respect to compute operations, meaning that we can start copying data while the GPU is still processing other data.
    This can help hide the latency of data transfer between CPU and GPU or between different GPUs.
    """

    # The first thing the function does is set the CUDA device to the specified cuda_id using torch.cuda.set_device().
    # It then creates a cpu_buf tensor with a size of 1 GB, which will be used as a temporary buffer to transfer data between the source and destination devices.
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    # The function then enters an infinite loop, where it continuously checks for items in the queue. If an item is None, it means that the function should exit,
    # so it returns. Otherwise, it extracts the source and destination tensors, along with their indices, from the item.
    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
