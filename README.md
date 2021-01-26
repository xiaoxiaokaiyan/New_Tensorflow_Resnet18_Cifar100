# 心得：**此代码可以很好地体现残差网络的构建**

## The complexity and accuracy of the neural network model


## Dependencies:
* Windows10
* python==3.6.12
* > GeForce GTX 1660TI
* tensorflow-gpu==2.0.0

## Visualization Results
![img1](https://github.com/xiaoxiaokaiyan/Protch_Transfer_learning_Pokmon/blob/main/batch.jpg)

* resnet101 训练结果 1

![img1](https://github.com/xiaoxiaokaiyan/New_Tensorflow_Resnet18_cifar100/blob/main/result_1.PNG)

* resnet101 训练结果 2

![img1](https://github.com/xiaoxiaokaiyan/New_Tensorflow_Resnet18_cifar100/blob/main/result_2.PNG)



## Public Datasets:

* cifar100

## Experience：
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
### 1.出现上面的错误
  * 原因一：You're out of memory
      * reducing your batch size
      * using a simpler model
      * using less data
      * limit TensorFlow GPU memory（本代码按此方法解决）： [https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in](https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in)(参考此网址)
          
```
      os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'**-----------------------------------（本代码按此方法解决）
       
      or
      
      physical_devices = tf.config.experimental.list_physical_devices('GPU')
      if len(physical_devices) > 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
```   
  * 原因二：You have incompatible versions of CUDA, TensorFlow, NVIDIA drivers, etc.
      * you can see [https://blog.csdn.net/qq_41683065/article/details/108702408](https://blog.csdn.net/qq_41683065/article/details/108702408)
        **my cudnn==7.6.4 cuda10.0_0  cudatoolkit==10.0.130**
        
 ### 2.tensorflow-gpu版本代码出现numpy错误
  * 其中一种解决方法：**pip install --upgrade numpy**
  

## References:
* 深度学习与TensorFlow 2入门实战（完整版）---龙曲良
