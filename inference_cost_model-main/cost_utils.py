hbm_efficiency = 0.5
comp_efficiency = 0.5
comm_efficiency = 0.5


def compute_model_memory_limit(num_layers, h_dim, b_type):
    return h_dim * h_dim * 12 * b_type * num_layers


def compute_prompt_time_stage(seq_in, batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree=1, h_dim=12288,
                              b_type=2) -> float:
    """
    The purpose of this function is to estimate the time taken to perform the initial computations required for the training of a deep learning model.
    It is an important parameter used in the calculation of the total time required to train the model.
    The function uses the various input parameters to calculate the time taken by the scan and compute stages of the prompt phase.

    The "prompt phase" is the initial phase of model training, where the inputs are forward-propagated through the model before the actual "training" begins.
    During this phase, the model's initial parameters are set, and the inputs are processed to produce the initial activations and gradients for each layer of the model.

    :param b_type: In the context of the compute_prompt_time_stage function, b_type refers to the "bits per element" used in the computation. It represents the number of bits required to represent each element or parameter of the model on the GPU.
    :return:
    """
    # layer_scan_time--load weight
    # layer_scan_time -- compute
    # 系数问题 是QKV，output（self-attention）以及MLP layer1，layer2的 参数
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / memory_bandwidth / hbm_efficiency
    # 系数问题，是针对Inference阶段，所以只考虑forward
    layer_compute_time = 24 * batch_size * seq_in * h_dim * h_dim / tp_degree / gpu_flops / comp_efficiency
    return (layer_scan_time + layer_compute_time) * num_layers


def compute_token_step_time_stage(batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree=1, h_dim=12288,
                                  b_type=2) -> float:
    layer_scan_time = 12 * h_dim * h_dim * b_type / tp_degree / memory_bandwidth / hbm_efficiency
    layer_compute_time = 24 * batch_size * h_dim * h_dim / tp_degree / gpu_flops / comp_efficiency
    return (layer_scan_time + layer_compute_time) * num_layers


def communicate_prompt_time_stage(seq_in, batch_size, num_layers, tp_degree, delay, bandwidth,
                                  h_dim=12288, b_type=2) -> float:
    """
    The communicate_prompt_time_stage function is used to estimate the time taken to communicate the initial model parameters during the prompt phase of deep learning model training.
    This involves the exchange of the initial weights and biases for each layer between different processors used for parallel model training.

    :param seq_in:
    :param batch_size:
    :param num_layers:
    :param tp_degree:
    :param delay:
    :param bandwidth:
    :param h_dim:
    :param b_type:
    :return:
    """
    step_time = 0
    for i in range(tp_degree):
        current_step = 0
        for j in range(tp_degree):
            if i != j:
                # is it used to communicate tensor?
                current_step += (delay+batch_size*seq_in*h_dim*b_type/tp_degree/bandwidth)
        # For each tensor core (tp_degree), the function computes the maximum time taken to communicate the model parameters between different tensor cores (current_step)
        # by computing the transfer time required to send and receive the model parameters for each layer of the neural network and accumulating the value in the current_step variable.
        step_time = max(step_time, current_step)
    # The step_time variable represents the maximum time taken to communicate the initial model parameters between different tensor cores for a single layer. Since there are num_layers in the deep learning model, the total time taken to communicate the initial parameters will be the sum of the time taken for each layer.
    #
    # The factor of 4 in the formula arises from the fact that there are 4 data exchanges between different processor groups involved in the parallel communication during the prompt phase of model training. These include:
    # Data exchange between the model replicas
    # Data exchange between the optimizer replicas
    # Data exchange between the forward signaling replicas
    # Data exchange between the backward signaling replicas
    result = step_time * 4 * num_layers / comm_efficiency
    return result


def communicate_token_step_time_stage(batch_size, num_layers, tp_degree, delay, bandwidth,
                                      h_dim=12288, b_type=2) -> float:
    step_time = 0
    for i in range(tp_degree):
        current_step = 0
        for j in range(tp_degree):
            if i != j:
                current_step += (delay + batch_size*h_dim*b_type/tp_degree/bandwidth)
        step_time = max(step_time, current_step)
    result = step_time * 4 * num_layers / comm_efficiency
    return result


def end_to_end_time(batch_size, seq_in, seq_out, memory_bandwidth, gpu_flops, delay, bandwidth, num_layers, tp_degree,
                    h_dim=12288, b_type=2) -> float:
    # The function is composed of several stages that calculate the time taken to complete the prompt phase,
    # which is the initial phase that sets up the model, and the token phase, which is where the actual training of the model is done.
    # The prompt_comp_time calculates the time taken to complete the compute stage of the prompt phase. It uses the compute_prompt_time_stage function,
    # which estimates the time taken to perform the computations required for the prompt phase on one or more GPUs.
    # This time is proportional to the number of layers in the model, the size of the inputs seq_in and batch_size, the memory bandwidth and flops of the GPU, num_layers, tp_degree, h_dim, and b_type.
    prompt_comp_time = compute_prompt_time_stage(seq_in, batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree,
                                                 h_dim, b_type)
    # The prompt_comm_time calculates the time taken to complete the communication stage of the prompt phase.
    # It uses the communicate_prompt_time_stage function, which estimates the time taken to transfer the input data to the GPUs and the results back to the host system.
    # This time is proportional to the seq_in, batch_size, num_layers, tp_degree, delay, bandwidth, h_dim, and b_type.
    prompt_comm_time = communicate_prompt_time_stage(seq_in, batch_size, num_layers, tp_degree, delay, bandwidth,
                                                     h_dim, b_type)

    print(f"Prompt phase time: {prompt_comp_time + prompt_comm_time}s (compute: {prompt_comp_time}s, "
          f"communication: {prompt_comm_time}s)")

    token_comp_time = compute_token_step_time_stage(batch_size, memory_bandwidth, gpu_flops, num_layers, tp_degree,
                                                    h_dim, b_type)
    token_comm_time = communicate_token_step_time_stage(batch_size, num_layers, tp_degree, delay, bandwidth,
                                                        h_dim, b_type)
    print(f"Token phase per-token time: {token_comp_time + token_comm_time}s (compute: {token_comp_time}s, "
          f"communication: {token_comm_time}s)")

    total_time = prompt_comp_time + prompt_comm_time + seq_out * (token_comp_time + token_comm_time)
    print(f"Total time: {total_time}s")
    print(f"Throughput: {batch_size*seq_out/total_time}")
    return total_time

