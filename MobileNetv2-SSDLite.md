
#  MobileNetv2-SSDLite 实现以及训练自己的数据集


### **1. 环境**
------

Caffe 实现 MobileNetv2-SSDLite 目标检测，预训练文件从 tensorflow 来的，要将 tensorflow 模型转换到 caffe.

先废话，我的环境，如果安装了 cuda, cudnn, 而且 caffe，tensorflow 都通过了，请忽略下面的，只是要注意 caffe 的版本：

- Ubuntu 16.04

- 查看 CUDA 版本：(CUDA 8.0.61)

```shell
alpha@zero:~/$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```

- 查看 cudnn 版本：(cudnn 6.0.21)

```shell
alpha@zero:~/$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR      6
#define CUDNN_MINOR      0
#define CUDNN_PATCHLEVEL 21
```

- caffe 版本，并非官方版本，因为官方版本的有很多网络的某些 layer 或者其他方面的不支持：

  1. caffe 版本：[weiliu89/caffe](https://github.com/weiliu89/caffe/tree/ssd)，如果用的这个版本，需要自己手动添加 ReLU6
  2. [chuanqi305 fork weiliu89](https://github.com/chuanqi305/ssd) 的版本，他添加了 ReLU6，（**推荐这个版本**）



- tensorflow 版本，因为我的 CUDA，cuDNN 版本的原因，我装了一个官方老版本 (tensorflow 1.3.0)。

```shell
alpha@zero:~/$ pip list | grep tensorflow
tensorflow-gpu (1.3.0)
tensorflow-tensorboard (0.1.8)
```



### **2. 具体实现**

- git clone MobileNetv2-SSDLite respository.

```shell
git clone https://github.com/chuanqi305/MobileNetv2-SSDLite
```



- 从 tensorflow 下载 MobileNetv2-SSDLite 的 tensorflow 模型到 **ssdlite/** 路径，并解压。

```shell
cd MobileNetv2-SSDLite/ssdlite/
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -zvxf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```



- 转换 tensorflow 模型到 caffe，执行下面两个脚本后，在当前目录下会生成 **deploy.caffemodel** caffe 模型

```shell
python dump_tensorflow_weights.py
# 修改 load_caffe_weight.py 中 caffe_root 为自己的路径
python load_caffe_weight.py  
```



- 默认生成的是 coco 数据集的 caffemodel 90 个类别(coco 数据集是80个类别，不清楚这里为什么有 90 个)，可以转换成 voc 模型，20 个类别，当然也可一改成自己数据集的 caffemodel，需要改 **coco2voc.py** 脚本，并执行：

```shell
# 修改 coco2voc.py 中 caffe_root 为自己的路径
python coco2voc.py
```



- 上面所有步骤成功了后，就可以 test 一下模型

  cd 到 MobileNetv2-SSDLite/ 目录下，**同样的修改脚本中 caffe_root 为自己的路径后**，执行下面命令：

```shell
python demo_caffe_voc.py
```

测试结果如下图片，

<div align=center><img height="400" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenetv2-ssdlite1.png"/></div>





### **3. 训练自己的数据集**

- **制作自己的数据集** 

  这两步请参考， https://blog.csdn.net/Chris_zhangrx/article/details/80458515

  MobileNet-SSD, MobileNetv2-SSDLite，训练时数据集需要做的准备一样。

- 修改 label 文件，在 MobileNetv2-SSDLite/ssdlite/voc/ 目录下 **labelmap_voc.prototxt，修改成自己的类别，例如有两类 car, person。注意 要加 background

```protobuf
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "car"
  label: 1
  display_name: "car"
}
item {
  name: "person"
  label: 2
  display_name: "person"
}
```



- 在 MobileNetv2-SSDLite/ssdlite/ 目录下的 **gen_model.py** 生成 caffe 训练测试时用的 prototxt 文件，注意 CLASS_NUM = 类别数 + 1，因为把背景也算进去，例如是上面描述的 car, person 两类，则 CLASS_NUM = 3，同样的 修改 caffe_root 为自己的路径

```shell
python gen_model.py -s train -c CLASS_NUM >train.prototxt
python gen_model.py -s test -c CLASS_NUM >test.prototxt
python gen_model.py -s deploy -c CLASS_NUM >deploy.prototxt
```

然后修改 train.prototxt 中的三处，

第一处，

```protobuf
source: "trainval_lmdb"
```

改成你自己的 trainval_lmdb 路径，例如

```shell
source: "/home/alpha/MobileNetv2-SSDLite/lmdb/trainval_lmdb"
```

第二处，

```shell
label_map_file: "labelmap.prototxt"
```

改成自己的 labelmap 路径，例如

```shell
label_map_file: "/home/alpha/MobileNetv2-SSDLite/ssdlite/voc/labelmap_voc.prototxt"
```

第三处，如果 GPU 很渣，训练时候超出显存，把 batch_size 改小一点

```protobuf
batch_size: 64
```





- 开始训练

  修改 MobileNetv2-SSDLite/ssdlite/voc/solver_train.prototxt ，或者复制到你自己的 project 目录下，其中

```protobuf
net: "train.prototxt"
```

  需修改成自己的 train.prototxt 路径，如下，或者复制到当前目录下

```protobuf
net: "/home/alpha/MobileNetv2-SSDLite/ssdlite/train.prototxt"
```

当然其他的参数都可以根据自己的训练修改，例如最大迭代次数，以及每隔多少次 保存一次模型，等等



修改并运行 MobileNetv2-SSDLite/ssdlite/voc/train.sh，例如

```shell
#!/bin/sh
mkdir -p snapshot
/home/alpha/ssd/build/tools/caffe train \
-solver="/home/alpha/MobileNetv2-SSDLite/ssdlite/voc/solver_train.prototxt" \
-weights="/home/alpha/MobileNetv2-SSDLite/ssdlite/deploy_voc.caffemodel" \
-gpu 0
```



如果顺利的话，god bless you! 就可以开始训练。 







### **others**
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com


