# 心得：**此代码可以很好地体现了自己数据的分类，既有自训练模型，也有预训练模型，resnet18和resnet101可以随时换着预训练**

## Dependencies:
* Windows10
* python==3.6.10
* > GeForce GTX 1660TI
* pytorch==1.0.0
* torchvision==0.2.1
* cuda100
* numpy==1.19.5

## Visualization Results

* resnet101 训练结果 1

![img1](https://github.com/xiaoxiaokaiyan/Protch_Transfer_learning_Pokmon/blob/main/resnet101%20test.PNG)

* resnet101 训练结果 2

![img1](https://github.com/xiaoxiaokaiyan/Protch_Transfer_learning_Pokmon/blob/main/resnet101%20test%20loss%20and%20acc.PNG)

* resnet101 训练结果 3

![img1](https://github.com/xiaoxiaokaiyan/Protch_Transfer_learning_Pokmon/blob/main/batch.jpg)





## Public Datasets:

* 五种pokemon




## Emphasize:
* pokemon图片在“宝可梦数据集.pdf”百度云盘，下载后文件夹解压出来直接运行程序即可，具体位置看“预训练模型放的位置.PNG”
* train_scratch.py文件是自己训练，train_transfer.py是使用resnet预训练模型
* transfer所使用的预训练模型可以随时更换，具体看“transfer所使用的预训练模型可以随时更换，具体看本图片.PNG”

## Experience：
### 1.出现下面的错误
  * 原因一：You're out of memory
      * reducing your batch size
      * using a simpler model
      * using less data
      * limit TensorFlow GPU memory：  <h3>os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'</h5>
    
    
    
    
```
Traceback (most recent call last):
  File "resnet18_train.py", line 89, in <module>
    main()
  File "resnet18_train.py", line 53, in main
    logits = model(x)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\keras\engine\base_layer.py", line 891, in __call__
    outputs = self.call(cast_inputs, *args, **kwargs)
  File "C:\Users\33203\Desktop\lesson43\resnet.py", line 72, in call
    x = self.stem(inputs)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\keras\engine\base_layer.py", line 891, in __call__
    outputs = self.call(cast_inputs, *args, **kwargs)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\keras\engine\sequential.py", line 270, in call
    outputs = layer(inputs, **kwargs)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\keras\engine\base_layer.py", line 891, in __call__
    outputs = self.call(cast_inputs, *args, **kwargs)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\keras\layers\convolutional.py", line 197, in call
    outputs = self._convolution_op(inputs, self.kernel)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\nn_ops.py", line 1134, in __call__
    return self.conv_op(inp, filter)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\nn_ops.py", line 639, in __call__
    return self.call(inp, filter)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\nn_ops.py", line 238, in __call__
    name=self.name)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\nn_ops.py", line 2010, in conv2d
    name=name)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\gen_nn_ops.py", line 1031, in conv2d
    data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\ops\gen_nn_ops.py", line 1130, in conv2d_eager_fallback
    ctx=_ctx, name=name)
  File "D:\installpackage\ana\envs\tf361\lib\site-packages\tensorflow_core\python\eager\execute.py", line 67, in quick_execute
    six.raise_from(core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [Op:Conv2D]
```

## References:
* 深度学习与TensorFlow 2入门实战（完整版）---龙曲良
