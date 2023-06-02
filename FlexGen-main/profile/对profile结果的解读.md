load weight -- reasonable

store cache / load cache -- reasonable，because of the sperate layer policy?

OPTLM的layers组成为

![image-20230525100729882](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\image-20230525100729882.png)

InputEmbed(1) + OutputEmbed(1) + selfAttention(num_hidden_layers) + MLP(num_hidden_layers)

layer 0 是InputEmbed ，layer 97是OutputEmbed layer1（奇数）是selfAttention layer2（偶数）是MLP



layer1 的compute time很长的原因：是prefill阶段，在self-attention计算过程中调用的是mha()函数，相比于之后的mha_gen()

1. **Input:** `mha()` takes in a single input vector per call, while `mha_gen()` takes in the entire sequence of input vectors at once.
2. **Purpose:** `mha()` is used during the encoding phase of the Transformer to compute attention for each input vector separately, while `mha_gen()` is used during the decoding phase to compute attention for the entire sequence at once.
3. **Caching:** `mha()` does not use caching, while `mha_gen()` uses cached key and value matrices for the entire sequence of input vectors to speed up attention calculation.
4. **Sparsity:** `mha_gen()` needs to consider sparsity, which is the proportion of attention weights that are zero.

# 对比dist 和 非dist

num_gpu_batches 4 gpu_batch_size_12 opt-35b percent 20 80 0 100 0 100

总时长 dist_opt 180s.  opt 270s

dist_opt：

node 0 耗时130s 4s的send时间，1秒的receive时间

node 1 耗时 153s 10s的起步receive时间 30s的send时间

node 2 耗时 166s 20s的起步receive时间 11s的send时间

node 3 耗时 180s  31s的起步receive时间 11s的send时间

opt:

compute 耗时132s   

load_weight和load_cache  分别耗时43 和 69s

prefill阶段两者耗时类似。但是到generation阶段，前馈（feedward层）compute耗时增加了很多，而self attention层基本一样



