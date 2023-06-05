import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.optim import AdamW, SGD
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
from timer import timers

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)


def custom_collate_fn(batch):
  # batch is a list of samples
  # each sample is a dictionary with keys "input_ids" and "labels"
  # input_ids is a tensor of shape (seq_len,)
  # labels is a tensor of shape (1,)

  # get the maximum sequence length in the batch
  max_len = max([len(sample["input_ids"]) for sample in batch])

  # create tensors to store the padded input_ids and labels
  padded_input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
  # padded_labels = torch.zeros(len(batch), 1, dtype=torch.long)

  # create a tensor to store the mask
  mask = torch.zeros(len(batch), max_len, dtype=torch.long)

  # iterate over the samples and copy the data to the tensors
  for i, sample in enumerate(batch):
    seq_len = len(sample["input_ids"])
    padded_input_ids[i, :seq_len] = torch.tensor(sample["input_ids"])
    # padded_labels[i] = torch.tensor(sample["labels"])
    mask[i, :seq_len] = 1

  # return the padded input_ids, labels and mask as a batch
  return {"input_ids":padded_input_ids, "attention_mask":mask}

def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
  )

# 设置训练参数
train_batch_size = 8 # train阶段批量大小
gradient_accumulation_steps = 1 # 梯度累积步数
max_steps = 200 # 最大训练步数
logging_steps = 1 # 日志记录步数间隔
inference_batch_size = 8 # inference阶段批量大小
output_dir = "outputs" # 输出目录
max_new_tokens = 50 # 设置每次生成文本的最大新标记数


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
data = load_dataset("Abirate/english_quotes")
# data.map is used to split the quote column in the dataset, and the batched=True parameter is set, which means that it is processing in batch mode, which can improve processing speed.
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
data_split = data['train'].train_test_split(test_size=0.1)
train_dataset = data_split['train']
inference_dataset = data_split['test']
# 创建数据加载器
inference_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=inference_batch_size, shuffle=True, collate_fn = custom_collate_fn)


print("++++++++++++len(train_dataset)++++++++++++++")
print(len(train_dataset))
print("++++++++++++len(inference_dataset)++++++++++++++")
print(len(inference_dataset))
# its weights in half-precision (float16) are about 13GB
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
     load_in_8bit = True,
     device_map = 'auto',
)

# 创建优化器
optimizer = SGD(model.parameters(), lr=5e-5)

# 创建损失函数
loss_fn = torch.nn.CrossEntropyLoss()
#  let's freeze all our layers, and cast the layer-norm in float32 for stability.
#  We also cast the output of the last layer in float32 for the same reasons.
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

# It allows the model to clear activation values for some layers as it propagates forward and recalculate them as it propagates back.
# This reduces memory usage, but adds some computation time.
# This function can only be used if the model supports gradient checkpoints and requires setting gradient_checkpointing to True in the model configuration.
model.gradient_checkpointing_enable()  # reduce number of stored activations
# a function that grads the input tensor of the model with a property of requires_grad=True
# so that the gradient of the input tensor can be calculated during backpropagation.
# This function can only be used if use_reentrant=True is used, otherwise there will be no gradient in the checkpoint portion of the model.
model.enable_input_require_grads()
model.lm_head = CastOutputToFloat(model.lm_head)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



# 开始训练循环
model.train() # 将模型设置为训练模式
total_loss = 0.0 # 累积的总损失值
step = 0 # 当前的训练步数

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=logging_steps,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
# 记录训练的开始时间
timers("train").reset()
timers("train").start()
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
timers("train").stop()
# 记录训练的结束时间，并计算时间消耗
costs = timers("train").costs

training_time = costs[0]

# 打印或保存前向传播的时间消耗
print(f"Training time: {training_time:.4f} s")

# 训练循环结束，打印结束信息
print("Training finished.")


# 开始推理循环
model.eval() # 将模型设置为推理模式

# 记录推理的开始时间
timers("inference").reset()
timers("inference").start()
batch_num = 0
with torch.cuda.amp.autocast():
  for batch in inference_dataloader:
    batch_num = batch_num + 1
    output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens)
    # print('\n\n', tokenizer.batch_decode(output_tokens, skip_special_tokens=True))

timers("inference").stop()
# 记录训练的结束时间，并计算时间消耗
inference_costs = timers("inference").costs
print(inference_costs)
inference_latency = inference_costs[0]

num_generated_tokens = len(inference_dataset)*max_new_tokens
# 打印或保存前向传播的时间消耗
print(f"inference_latency: {inference_latency:.4f} s")
# 打印吞吐量
print(f"inference_throughput: {num_generated_tokens/inference_latency:.4f} token/s")
# 训练循环结束，打印结束信息
print("inference finished.")
