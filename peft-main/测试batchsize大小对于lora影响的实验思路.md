老师好，我感觉这个文件中的任务挺合适的，使用mt0-large，参数个数为1.2billion，有finetuning也有training阶段，老师上次meeting说的是否类似于prefilling问题，我思考了下，觉得finetuning阶段，每一次epoch，在反向传播之前确实是类似prefill的，所以我准备在代码的这里记时，

![image-20230603115003432](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\image-20230603115003432.png)然后算出平均latency和throughput。这一步的batchsize调整可以直接调整前面代码的参数，然后就会直接作用于数据集，很方便。

![image-20230603115035243](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\image-20230603115035243.png)

![image-20230603115014270](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\image-20230603115014270.png)

同时代码中也写好了inference的代码,我准备在这里计时，

![image-20230603115050315](C:\Users\kevin\AppData\Roaming\Typora\typora-user-images\image-20230603115050315.png)同时inputs的num和output的length也很好取得，算total_throughput很方便(这一步我不准备区分prefill和token generation阶段了,不太方便在model.generate()方法中增加计时操作)

至于调整batchsize我想的是调整数据集的取样为下面这样

```
inputs = tokenizer(dataset[“validation”][text_column][i:i+batchsize], padding=True, truncation=True, return_tensors=“pt”)
```