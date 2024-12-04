# 遇到的问题：

![image-20230516114939276](prefill任务遇到的问题.assets/image-20230516114939276.png)

原因，驱动版本太低：

nvida-smi表示目前显卡驱动最高支持的cuda版本 不代表你已经安装了cuda

![image-20230516120234985](prefill任务遇到的问题.assets/image-20230516120234985.png)

解决方式，安装cuda 11.6

![image-20230516115012944](prefill任务遇到的问题.assets/image-20230516115012944.png)

python3 -m prefill.dist_flex_opt_prefill --model facebook/opt-1.3b --gpu-batch-size 16  --percent 100 0 100 0 100 0  --comm-device gpu

![image-20230516114819581](prefill任务遇到的问题.assets/image-20230516114819581.png)

![image-20230516114829087](prefill任务遇到的问题.assets/image-20230516114829087.png)





![image-20230516122159264](prefill任务遇到的问题.assets/image-20230516122159264.png)

![image-20230516122208461](prefill任务遇到的问题.assets/image-20230516122208461.png)