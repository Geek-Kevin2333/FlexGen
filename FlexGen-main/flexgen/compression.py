import dataclasses

import torch
import numpy as np

from flexgen.pytorch_backend import (TorchTensor, TorchDevice,
    DeviceType, general_copy, fix_recursive_import)
from flexgen.utils import np_dtype_to_torch_dtype


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


class TorchCompressedDevice:
    """Manage tensors stored in a compressed format."""

    def __init__(self, base_device):
        self.name = "compressed"
        self.device_type = DeviceType.COMPRESSED
        self.base_device = base_device

        self.data_decompress_workspace = None
        self.workspace_pt = 0

    def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
        """Allocate a compressed TorchTensor. Round up the shape to group boundary."""
        assert comp_config.num_bits == 4 and dtype == np.float16

        group_size, group_dim = comp_config.group_size, comp_config.group_dim

        # Round up
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])

        data = self.base_device.allocate(data_shape, np.uint8, pin_memory=pin_memory)
        scale = self.base_device.allocate(scale_shape, np.float16, pin_memory=pin_memory)

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (data, scale, comp_config), self, name=name)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        return k_cache, v_cache
    # 这段代码是一个方法，用于在初始化时为执行 attention compute（注意力计算）过程时需要的 fp32（单精度浮点数）工作区分配内存。
    # 该方法接收三个参数，分别是config、task和policy，其中config包含了与模型架构相关的参数，task包含了任务相关的参数，policy包含了一些策略参数。
    # 首先，该方法会检查当前的设备类型是否是CPU，如果不是，则直接返回。接着，它会从policy中获取gpu_batch_size、comp_cache_config等参数，并根据config和task计算出其他需要的参数。
    # 然后，该方法会计算出需要的内存空间大小，并使用torch.empty()函数为两个data_decompress_workspace元素分别分配空间。这两个元素分别用于缓存经过解压缩后的键值矩阵和数值矩阵，它们是在执行attention compute过程中需要的两个重要的输入矩阵。注意，这段代码的主要目的是在CPU上为attention compute过程分配内存空间。
    # Regenerate response
    def init_attention_compute_workspace(self, config, task, policy):
        if self.base_device.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        b = policy.gpu_batch_size
        n_head = config.n_head
        head_dim = config.input_dim // n_head
        max_seq_len = task.prompt_len + task.gen_len - 1
        shape = (max_seq_len, b * n_head, head_dim)

        group_size, group_dim = (
            policy.comp_cache_config.group_size, policy.comp_cache_config.group_dim)
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])

        self.data_decompress_workspace = [
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
        ]

    def compress(self, tensor, comp_config):
        """Compress a torch.Tensor. Round up the shape to group boundary."""
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)
        assert num_bits == 4 and group_size % 2 == 0 and not symmetric
        # The purpose of the sentence is to check if the device type of the tensor object is CPU and its data type is float16.
        # If both conditions are true, it converts the tensor object to float type.
        # This conversion is needed to ensure compatibility with the quantization process later in the function.
        # The quantization process in the function requires values to be first converted to float data type before being scaled and quantized.
        if tensor.device.type == "cpu" and tensor.dtype == torch.float16:
            tensor = tensor.float()

        shape = tensor.shape
        num_groups = (shape[group_dim] + group_size - 1) // group_size

        # Pad
        # The purpose of the following code block is to prepare the tensor for the quantization process by padding and reshaping the tensor data into the required shape:
        # Finally, the tensor data is reshaped into the new shape and returned as the data object. The data object will further be processed for quantization to save memory.
        # todo： If I am gonna converse the data form，how can I know what form is needed？
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])
        pad_len = (group_size - shape[group_dim] % group_size) % group_size
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            tensor = torch.cat([
                tensor,
                torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
                dim=group_dim)
        data = tensor.view(new_shape)

        # Quantize the algorithm showed in the paper
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_().to(torch.uint8)

        # Pack
        # This part is responsible for packing the tensor data into a smaller data representation.
        # The packed tensor represents the original tensor data in a smaller number of bits to save space.
        # The algorithm packs pairs of data points into a single byte.
        # To accomplish this, the code constructs two tuples of slice objects - one for the even data points,
        # and one for the odd data points.
        # The even-indexed elements are shifted left by 4 bits to make room for the odd-indexed points,
        # then 'or'ed to combine the two indices into a single uint8 value.
        # In simpler words, this step packs 2 consecutive values to a single byte by shifting one value by 4 bits
        # and combing that with the second value to be part of a single byte.

        # group_dim means that the dimension corresponds to the groups that will be packed
        # and quantized together in the subsequent steps.
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))
        data = torch.bitwise_or(
            data[left_indices].bitwise_left_shift(4), data[right_indices])

        # Reshape
        # This part of the code has two sections, one for reshaping the packed tensor (data)
        # and one for the scale factor tensor.
        # The data tensor is reshaped to the size that was calculated in the previous code block.
        # The number of rows is the same as the number of groups that were calculated,
        # while the number of columns is equal to the product of group size and 0.5 (as two data points are packed into one byte).
        # This result is the first element of the returned tuple.
        # The scale tensor underwent a similar reshaping process to become compatible with the data tensor.


        # an example to elaborate on the the following codes
        # Suppose we have a 3-dimensional PyTorch Tensor tensor with shape (2, 3, 4), and we set group_dim=1.
        # This means that the dimension 1 corresponds to the groups that will be packed and quantized together in the subsequent steps.
        # Now, let data be the PyTorch tensor we want to extract slices from.
        # The tuple comprehension in the first line starts with a slice object that extracts the first group_dim+1 dimensions from data.
        # In this case, we take the first two dimensions of data (0 and 1), since group_dim is 1 and we add 1 to it. As a result, left_indices contains the following tuple:
        # Notice that the last slice object (0, 4, 2) selects every other index in the third dimension of data.
        # This is because we want to pack pairs of data points together, so we need to take every other element in the specified slice to get the even elements.
        # The odd elements will be selected in right_indices with a slice starting from the index 1.
        # In the subsequent step of the code block, data_left_indices is shifted left by 4 bits using the bitwise_left_shift() function,
        # combined with the odd-indexed elements using bitwise_or(), and finally packed into single 8-bit uints as needed for quantization.
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])
        data = data.view(data_shape)
        scale = torch.cat([scale, mn], dim=group_dim+1).view(scale_shape)

        data = TorchTensor.create_from_torch(data, self.base_device)
        scale = TorchTensor.create_from_torch(scale, self.base_device)

        return TorchTensor(shape, tensor.dtype,
                           (data, scale, comp_config), self)

    def decompress(self, tensor):
        data, scale, comp_config = tensor.data
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)

        group_size_c = group_size // 2
        shape = data.shape
        num_groups = (shape[group_dim] + group_size_c - 1) // group_size_c

        # Pad
        new_shape = (shape[:group_dim] + (num_groups, group_size_c) +
                     shape[group_dim+1:])
        pad_len = (group_size_c - shape[group_dim] % group_size_c) % group_size_c
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            data = torch.cat([
                data,
                torch.zeros(pad_shape, dtype=data.dtype, device=data.device)],
                dim=group_dim)
        packed = data.data.view(new_shape)

        # Unpack
        if self.base_device.device_type == DeviceType.CPU:
            self.workspace_pt = (self.workspace_pt + 1) % len(
                self.data_decompress_workspace)
            data = self.data_decompress_workspace[
                self.workspace_pt][:shape[0]]
        else:
            new_shape = (shape[:group_dim] + (num_groups, group_size,) +
                         shape[group_dim+1:])
            data = torch.empty(new_shape, dtype=torch.float16, device=packed.device)
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))
        data[left_indices] = packed.bitwise_right_shift(4)
        data[right_indices] = packed.bitwise_and(0xF)

        # Dequantize
        scale, mn = scale.data.split(1, dim=group_dim + 1)
        data.div_(scale)
        data.add_(mn)

        # Reshape
        unpad_len = (group_size - tensor.shape[group_dim] % group_size) % group_size
        if unpad_len != 0:
            flatten_shape = (shape[:group_dim] + (num_groups * group_size,) +
                             shape[group_dim+1:])
            indices = [slice(0, x) for x in flatten_shape]
            indices[group_dim] = slice(0, flatten_shape[group_dim] - unpad_len)
            data = data.view(flatten_shape)[indices].contiguous()

        return data.view(tensor.shape)
# The compressed tensor divides the data tensor into fixed-length subsequences, each of length group_size, along a dimension group_dim.
# When the data is reconstructed from the compressed tensor, the compression groups are first split into their original length of group_size and rescaled using a common scaling factor.
# Then, these values are combined into individual tensor elements.

# While copying data, the slice indices passed to the general_copy() function may not perfectly align with these compression groups
# because they are specified independently of the compression.
def general_copy_compressed(dst, dst_indices, src, src_indices):
    assert (src.device.device_type == DeviceType.COMPRESSED and
            dst.device.device_type == DeviceType.COMPRESSED)

    src_data_indices, src_scale_indices = get_compressed_indices(
        src, src_indices, src.shape)

    dst_data_indices, dst_scale_indices = get_compressed_indices(
        dst, dst_indices, dst.shape)

    general_copy(dst.data[0], dst_data_indices, src.data[0], src_data_indices)
    general_copy(dst.data[1], dst_scale_indices, src.data[1], src_scale_indices)


def get_compressed_indices(tensor, indices, shape):
    """
    This code defines the get_compressed_indices() function which is used to retrieve indices of the compressed tensor data and scale tensors.
    @param tensor: the tensor object being compressed
    @param indices: a tuple of slice objects representing the indices of the tensor to compress
    @param shape: the output shape of the compressed tensor.
    @return:
    """
    # The data attribute of the tensor object contains the three parameters that define how the tensor was compressed: group_size, group_dim, and num_bits
    comp_config = tensor.data[2]
    group_size, group_dim = comp_config.group_size, comp_config.group_dim
    assert comp_config.num_bits == 4
    # The code first checks if the indices argument is set to None. If so, indices is set to represent the full tensor in slice notation by taking the first (group_dim + 1) dimensions of the tensor.
    if indices is None:
        indices = list(slice(0, x) for x in shape[:group_dim+1])
    # If indices are specified, the function pads indices with slice(0, x) objects
    # to cover any unspecfied dimensions in the returned compressed tensor shape.
    else:
        indices = list(indices) + [slice(0, x) for x in shape[len(indices):]]
    # The step that checks if the starting index of the compression group of group_size elements in the group_dim dimension is a multiple of group_size is important to
    # ensure that the compressed tensor can be properly reconstructed.
    # In order to compress the tensor, it is first split into fixed-length subsequences, each of length group_size, along the group_dim dimension.
    # These subsequences are then compressed using a common scale factor and represented using a lower number of bits.
    # When reconstructing the tensor, this procedure is reversed and the compressed tensor is first split into fixed-length subsequences for which a scale factor is shared,
    # and the bits are then combined and decompressed to obtain the original tensor data.
    assert indices[group_dim].start % group_size == 0
    # retrieves the appropriate indices for the compressed data and the scale tensors. This is done by creating new tuples data_indices and scale_indices.
    data_indices = list(indices)
    # The data_indices tuple is then modified by replacing the slice object at index group_dim with a new slice object that limits the range of indices to be retrieved from the compressed tensor. Specifically, it halve the range of indices to account for the fact that each value is represented using 4 bits rather than the original 8 bits,
    # thereby reducing the number of elements by a factor of two.
    data_indices[group_dim] = slice(
        indices[group_dim].start // 2, (indices[group_dim].stop + 1) // 2)

    scale_indices = indices
    scale_indices.insert(group_dim+1, slice(0, 2))
    # The slice object at index group_dim in the scale_indices tuple is then replaced with a new slice object corresponding to the range of indices that should be retrieved from the scale tensor.
    # Here, this slice object is represented using group_size to ensure that it covers the correct range of elements in the compressed tensor.
    # The scale tensor always has two values per compression group - one to scale the non-zero values and another to scale the zeroes.
    scale_indices[group_dim] = slice(
        indices[group_dim].start // group_size,
        (indices[group_dim].stop + group_size - 1) // group_size)

    return data_indices, scale_indices


default_cache_config = CompressionConfig(
    num_bits=0, group_size=0, group_dim=0, symmetric=False, enabled=False)


def set_cache_compression_config(config):
    global default_cache_config
    default_cache_config = config


def get_cache_compression_config():
    return default_cache_config


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim+1:])

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] +
            (original_shape[group_dim] + pad_len,) +
            original_shape[group_dim+1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def compress_and_decompress(tensor, config):
    packed_data = compress(tensor, config)
    return decompress(packed_data, config)


def test_simulated_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (64, 64, 64), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed_data = compress(a, config)
    b = decompress(packed_data, config)
    print(a[0])
    print(b[0])


def test_real_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (32, 1, 1), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    dev = TorchDevice("cuda:0", 0, 0).compressed_device
    packed = dev.compress(a, config)
    b = dev.decompress(packed)

    print(a.flatten())
    print(b.flatten())


if __name__ == "__main__":
    fix_recursive_import()
    #test_simulated_compression()
    test_real_compression()
