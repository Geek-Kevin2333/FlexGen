from cost_utils import *


GPU_spec = {}
# HBM: GB/s, dRAM: GB
GPU_spec["A100-80G"] = {"TFLOPS": 312, "HBM": 1935, "dRAM": 80}
GPU_spec["RTX-Ada-6000-48G"] = {"TFLOPS": 364, "HBM": 960, "dRAM": 48}
GPU_spec["A100-40G"] = {"TFLOPS": 312, "HBM": 1555, "dRAM": 40}
GPU_spec["RTX-4090"] = {"TFLOPS": 380, "HBM": 1000, "dRAM": 24}
GPU_spec["A6000"] = {"TFLOPS": 149.7, "HBM": 768, "dRAM": 48}
GPU_spec["A40"] = {"TFLOPS": 149.7, "HBM": 696, "dRAM": 48}
GPU_spec["A30"] = {"TFLOPS": 165, "HBM": 933, "dRAM": 24}
GPU_spec["V100-32GB"] = {"TFLOPS": 125, "HBM": 900, "dRAM": 32}
GPU_spec["RTX-3090-Ti"] = {"TFLOPS": 142.5, "HBM": 1008, "dRAM": 24}
GPU_spec["RTX-3090"] = {"TFLOPS": 142.5, "HBM": 935, "dRAM": 24}
GPU_spec["RTX-Quadro-6000"] = {"TFLOPS": 130.5, "HBM": 672, "dRAM": 24}
GPU_spec["RTX-Titan"] = {"TFLOPS": 130.5, "HBM": 672, "dRAM": 24}
GPU_spec["A10"] = {"TFLOPS": 125, "HBM": 600, "dRAM": 24}
GPU_spec["A5000"] = {"TFLOPS": 91, "HBM": 768, "dRAM": 24}
GPU_spec["V100-16GB"] = {"TFLOPS": 125, "HBM": 900, "dRAM": 16}
GPU_spec["RTX-3080-Ti"] = {"TFLOPS": 136.4, "HBM": 912, "dRAM": 12}
GPU_spec["A4500"] = {"TFLOPS": 94.5, "HBM": 640, "dRAM": 20}
GPU_spec["RTX-3080-12G"] = {"TFLOPS": 122, "HBM": 912, "dRAM": 12}
GPU_spec["RTX-3060-Ti-10G"] = {"TFLOPS": 129.6, "HBM": 448, "dRAM": 12}
GPU_spec["A4000"] = {"TFLOPS": 76.7, "HBM": 448, "dRAM": 16}
GPU_spec["T4"] = {"TFLOPS": 65.2, "HBM": 300, "dRAM": 16}
GPU_spec["GTX-Titan-V"] = {"TFLOPS": 110.0, "HBM": 651, "dRAM": 12}
GPU_spec["RTX-3060-12G"] = {"TFLOPS": 101.9, "HBM": 360, "dRAM": 12}
GPU_spec["RTX-2080-Ti"] = {"TFLOPS": 107.6, "HBM": 616, "dRAM": 11}
GPU_spec["RTX-3060-Ti-8G"] = {"TFLOPS": 129.6, "HBM": 608, "dRAM": 8}
GPU_spec["RTX-2080-Super"] = {"TFLOPS": 89.2, "HBM": 496, "dRAM": 8}
GPU_spec["RTX-3070-Ti"] = {"TFLOPS": 87, "HBM": 608, "dRAM": 8}
GPU_spec["RTX-3070"] = {"TFLOPS": 81, "HBM": 448, "dRAM": 8}
GPU_spec["RTX-2080"] = {"TFLOPS": 80.5, "HBM": 448, "dRAM": 8}
GPU_spec["RTX-3050"] = {"TFLOPS": 72.8, "HBM": 224, "dRAM": 8}
GPU_spec["RTX-2070-Super"] = {"TFLOPS": 72.5, "HBM": 448, "dRAM": 8}
GPU_spec["RTX-2070"] = {"TFLOPS": 59.7, "HBM": 448, "dRAM": 8}
GPU_spec["RTX-2060-Super"] = {"TFLOPS": 57.4, "HBM": 448, "dRAM": 8}
GPU_spec["RTX-2060"] = {"TFLOPS": 41.9, "HBM": 336, "dRAM": 6}


model_spec = {}
model_spec["gpt-j-6b"] = {"model_dim": 4096, "b_type": 2, "num_layer": 28}
model_spec["gpt-neox-20b"] = {"model_dim": 6144, "b_type": 2, "num_layer": 44}

connect_spec = {}
# bandwidth: GB/s
connect_spec["PCIe1.0X1"] = {"delay": 0, "bandwidth": 0.25}
connect_spec["PCIe1.0X4"] = {"delay": 0, "bandwidth": 1}
connect_spec["PCIe2.0X1"] = {"delay": 0, "bandwidth": 0.5}
connect_spec["PCIe3.0X1"] = {"delay": 0, "bandwidth": 1}
connect_spec["PCIe3.0X4"] = {"delay": 0, "bandwidth": 4}
connect_spec["PCIe4.0X16"] = {"delay": 0, "bandwidth": 32} # The current Crusoe A100-80G setting


def check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe2.0X1", batch_size=1, seq_in=128, seq_out=32):
    """
    :param model_key: The name of the deep learning model to be evaluated (default is "gpt-j-6b").
    :param connect_key: The type of intra-node connection used (default is "PCIe2.0X1").
    :param batch_size: The batch size of the data used for model training (default is 1).
    :param seq_in: The length of the input sequence used for model training (default is 128).
    :param seq_out: The length of the output sequence used for model training (default is 32).
    """
    with open(f"./{model_key}_{connect_key}_b{batch_size}_in{seq_in}_out{seq_out}.md", "w") as fp:
        fp.write("# Cost Estimation\n")
        fp.write("### Setting\n")
        # Within the function, it calculates several values such as the number of layers in the model,
        # the model size, the bandwidth of the intra-node connection, and the memory bandwidth and size of different types of GPUs.
        # Using these values, it estimates the time and throughput for each GPU type to train the model.
        num_layers = model_spec[model_key]['num_layer']
        h_dim = model_spec[model_key]['model_dim']
        b_type = model_spec[model_key]['b_type']
        model_size = compute_model_memory_limit(num_layers, h_dim, b_type)
        delay = 0
        bandwidth = connect_spec[connect_key]["bandwidth"] * 1073741824
        print(f"Model <{model_key}>, model_size: {model_size/1073741824} GB\n")
        # It outputs a table in Markdown format which includes the following information for each GPU type:
        # GPU-type: The type of GPU being evaluated.
        # TFLOPS: The theoretical computing power of the GPU in teraflops.
        # HBM: The size of the GPU's high-bandwidth memory in gigabytes.
        # dRAM: The size of the GPU's dynamic random access memory in gigabytes.
        # tp degree: The number of GPUs required to run this model with the given settings.
        # total-time (s): The estimated total time to run the model on the GPU in seconds.
        # throughput (token/s): The estimated throughput of the model in tokens (a measure of the total number of words processed) per second.
        fp.write(f"- Model: {model_key} (model dim: {h_dim}, num of layers: {num_layers})\n")
        fp.write(f"- Intra-node connection: {connect_key}\n")
        fp.write(f"- Batch size: {batch_size}\n")
        fp.write(f"- Input sequence length: {seq_in}\n")
        fp.write(f"- Output sequence length: {seq_out}\n\n")
        fp.write("| GPU-type  | TFLOPS  | HBM | dRAM  | tp degree | total-time (s) | throughput (token/s) |\n")
        fp.write("|----|----|----|----|----|----|----|\n")
        for gpu_type in GPU_spec.keys():
            # Within each iteration of the loop, the function extracts the HBM, dRAM, and TFLOPS values for the current gpu_type.
            # These values are then used to calculate the number of GPUs needed to run the model in parallel, which is stored as tp_degree.
            # HBM is designed to provide high-bandwidth, low-latency access to memory, which is useful for high-performance workloads such as deep learning and scientific computing.
            # Like dRAM, it uses capacitors to store data, but HBM stacks these capacitors in three dimensions to increase their density and reduce the distance that data must travel,
            # resulting in higher bandwidth and lower power consumption.
            #
            # HBM memory is usually located on the same package as the processor, which means that it has very low latency and can transfer data directly without going through the motherboard.
            # This feature makes it very suitable for high-performance applications that require fast data access, such as machine learning, rendering or gaming workloads.
            memory_bandwidth = GPU_spec[gpu_type]['HBM'] * 1073741824

            # dRAM stands for Dynamic Random-Access Memory. It is a type of computer memory
            # that is commonly used as the main memory in computing systems such as personal computers, mobile devices, and servers.
            # dRAM is called "dynamic" because it needs to be constantly refreshed to retain data.
            # It uses capacitors to store and access data, and these capacitors must be constantly recharged to maintain their data.
            # dRAM is faster than hard disk drives, but slower than cache memory or SRAM (Static RAM).
            memory_limit = GPU_spec[gpu_type]['dRAM'] * 1073741824
            gpu_flops = GPU_spec[gpu_type]['TFLOPS'] * 1e12
            tp_degree = 1
            # tp_degree is a variable that represents the number of GPUs required to run a specific deep learning model with the given settings.
            # The value of tp_degree is based on an assumption that each GPU can handle a certain amount of data before reaching its memory limitations.
            # In the code, tp_degree is calculated such that the combined memory capacity of all the GPUs used is at least 1.5 times the size of the model.
            while tp_degree * memory_limit < model_size * 1.5:
                tp_degree *= 2
            print(f"GPU <{gpu_type}>, tp: {tp_degree}")
            total_time = end_to_end_time(batch_size, seq_in, seq_out, memory_bandwidth, gpu_flops, delay, bandwidth,
                                         num_layers, tp_degree, h_dim, b_type)
            throughput = batch_size * seq_out / total_time
            fp.write(f"|{gpu_type}|{GPU_spec[gpu_type]['TFLOPS']}|{GPU_spec[gpu_type]['HBM']}|"
                     f"{GPU_spec[gpu_type]['dRAM']}|{tp_degree}|{total_time:.3f}|{throughput:.3f}|\n")


check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe2.0X1", batch_size=1, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe2.0X1", batch_size=16, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe2.0X1", batch_size=1, seq_in=512, seq_out=256)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe2.0X1", batch_size=16, seq_in=512, seq_out=256)

check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe2.0X1", batch_size=1, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe2.0X1", batch_size=16, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe2.0X1", batch_size=1, seq_in=512, seq_out=256)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe2.0X1", batch_size=16, seq_in=512, seq_out=256)

check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe4.0X16", batch_size=1, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe4.0X16", batch_size=16, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe4.0X16", batch_size=1, seq_in=512, seq_out=256)
check_model_for_all_gpus(model_key="gpt-j-6b", connect_key="PCIe4.0X16", batch_size=16, seq_in=512, seq_out=256)

check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe4.0X16", batch_size=1, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe4.0X16", batch_size=16, seq_in=128, seq_out=32)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe4.0X16", batch_size=1, seq_in=512, seq_out=256)
check_model_for_all_gpus(model_key="gpt-neox-20b", connect_key="PCIe4.0X16", batch_size=16, seq_in=512, seq_out=256)