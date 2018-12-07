
#  YOLO，you know?


### **1. 前言**
------

YOLOv1 早已过时，但历史总是重要的，为了完整性，还是记录之前学习的基础，以便总结改进算法时能清晰的定位改进点，学习优化的思想。

**此外在深度学习之前**，目标检测领域最好的算法就是 **DPM**（Deformable parts models），简单提一下：

- 利用了 SIFT 特征等，通过滑动窗口（sliding window）在图像中提出目标区域，然后 SVM 分类器来实现识别。
- 针对目标物体多视角问题，采用多组件（Component）策略
- 针对目标物体本身的形变，采用基于图结构（Pictorial Structure）的部件模型策略



后来，在深度学习的帮助下，目标检测任务有了一些进步。对于二维目标检测，主要任务为：

- **识别**，识别图片中的目标为何物
- **定位**，确定目标在图片中的位置，输出预测的边界框



结合识别精度和定位精度，在目标检测中用综合评价指标 mAP 描述模型的好坏；此外，在应用中必须考虑 detection 的实时性，模型在做 detection 时的处理器性能有限，不同于训练时可以用较好的显卡；但是往往 mAP 和实时性在一定程度上，存在对立，举个例子，对于同类型的深度神经网络结构，如果深度越深，一般模型最后达到的 mAP 更高，但深度越深，网络参数越多，实时性降低，有时候就需要**在 mAP 和 FPS 之间 trade-off** 一下，如果能同时的提升这两个指标，那就 nice 了。YOLO 网络出来的时候，精度不能算顶尖的，但是实时性绝对数一数二，对于很多追求实时性的应用是不错的选择，YOLO 在单 Pascal Titan X GPU上达到 **45 FPS** ，Tiny-yolo **150FPS**，在改进的 YOLOv2, YOLOv3 实时性上稍有提升，在精度上有较大提升，成功挤进几个优秀的目标检测算法行列。



基于深度学习的目标检测算法可以分为两类：

- **Two stage Detection**，以 **RCNN** 为代表，第一阶段由 **selective search** 生成大量 **region proposals**，即边界框，将取得的这些 proposal 通过 CNN （文中用的 AlexNet）来提取特征。第二阶段，分类器对上一阶段生成的边界框内的子图像分类，当然 RCNN 也有升级版的 Fast RCNN 和 Faster RCNN。
- **Unified Detection**，以 YOLO 和 SSD 为代表，YOLO 直接采用 regression 的方法进行 bbox 的检测以及分类，使用一个 **end-to-end** 的简单网络，直接实现坐标回归与分类，如论文中的原图：

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo1.png"/></div>



### **2. YOLO 如何实现**
----
- input 一幅图像，分成 **SxS** 个网格（grid cell），每个 grid cell 边长为单位 1，如果某个 object 的中心落在这个 grid cell 中，该 grid cell 就负责预测这个 object

- 每个 grid cell 要预测 **B** 个 bboxes，及其**置信度（confidence）**。


```c
# Pr(object) = 1 bbox 包含 object， Pr(object) = 0 bbox 只包含背景
Pr(object) = bbox 的可能性

# ground-truth 和 predicted bbox 框框的交并比
IOU(truth,pred) = 预测框与实际框的 IOU  

# 定义 confidence 如下，因此置信度包含识别和定位两方面
define confidence = Pr(object)*IOU(truth,pred)
```
> 补充1：如何判断一个 grid cell 中是否包含 object 呢？ 既一个 object 的 ground truth 的中心点坐标在一个grid cell中。
>
> 补充2：如果 grid cell 包含 object 中心时， Pr(object) = 1，那么 confidence = IOU(truth,pred)

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo2.png"/></div>






- 此外，YOLO 预测出的 bbox 包含 5 个值 （x, y, w, h, c），下图给出 S=3，图像 size 为 448x448 的例子。

```c
(x, y) = bbox 的中心相对于 grid cell 左上角顶点的偏移值 （wrt. grid cell）
(w, h) = bbox 的归一化宽和高 （wrt. image）
c = confidence，等价于 IOU(truth,pred)
```

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo3.png"/></div>





- 每一个 grid cell 要预测出 **C** 个类别概率值，表示该 grid cell 在包含 object 的条件下属于某个类别的概率，即 Pr(class-i|object），测试时将该条件概率乘上每个 bbox 的 confidence，得到每个 bbox 属于**某个类别的 conf.score**（class-specific confidence score）。

```c
class specific confidence score = Pr(classi|object)*confidence
								= Pr(classi|object)*Pr(object)*IOU(truth,pred)
                                = Pr(classi|object)*IOU(truth,pred)
```



- 将上述所有预测的值 encode 到 **tensor**。

```c
tensor 的数量 = SxSx(B*5+C)
```

> 一幅图像 SxS 个 grid cell，每个 grid cell 预测 B 个 bbox，每个 bbox 包含 5 个值 (x,y,w,h,c)，这样就有 SxSxB*5 个值;
>
> 另外，每个 grid cell 预测 C 个概率值，这样一幅图像就有 SxSxC 个值概率值；
>
> 如下图是 S=3，B=2，C=3 的时候

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo4.png"/></div>



### **3. YOLO 网络结构**
----
YOLO 的网络结构看起来比较简单，就像最基础的 CNN结构，卷积层，max 池化层，激活层，最后全连接层。

```c
┌────────────┬────────────────────────┬───────────────────┐
│    Name    │        Filters         │ Output Dimension  │
├────────────┼────────────────────────┼───────────────────┤
│ Conv 1     │ 7 x 7 x 64, stride=2   │ 224 x 224 x 64    │
│ Max Pool 1 │ 2 x 2, stride=2        │ 112 x 112 x 64    │
│ Conv 2     │ 3 x 3 x 192            │ 112 x 112 x 192   │
│ Max Pool 2 │ 2 x 2, stride=2        │ 56 x 56 x 192     │
│ Conv 3     │ 1 x 1 x 128            │ 56 x 56 x 128     │
│ Conv 4     │ 3 x 3 x 256            │ 56 x 56 x 256     │
│ Conv 5     │ 1 x 1 x 256            │ 56 x 56 x 256     │
│ Conv 6     │ 1 x 1 x 512            │ 56 x 56 x 512     │
│ Max Pool 3 │ 2 x 2, stride=2        │ 28 x 28 x 512     │
│ Conv 7     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 8     │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 9     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 10    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 11    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 12    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 13    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 14    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 15    │ 1 x 1 x 512            │ 28 x 28 x 512     │
│ Conv 16    │ 3 x 3 x 1024           │ 28 x 28 x 1024    │
│ Max Pool 4 │ 2 x 2, stride=2        │ 14 x 14 x 1024    │
│ Conv 17    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 18    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 19    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 20    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 21    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 22    │ 3 x 3 x 1024, stride=2 │ 7 x 7 x 1024      │
│ Conv 23    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ Conv 24    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ FC 1       │ -                      │ 4096              │
│ FC 2       │ -                      │ 7 x 7 x 30 (1470) │
└────────────┴────────────────────────┴───────────────────┘
```



对于这个网络结构，**需注意到的一些点**：

- 为什么最后的输出 size 为 7x7x30 ？因为文中 设置 S=7,B=2,C=20。SxSx(B*5+C)。那么我们需要调整到一个不同的 grid size 时，或者说我们的数据集 class 数量改变时，这时就需要调整 layer 的维度。
- 这个是 YOLO 的 full 版本，如果想追求更高的检测速度，可以适当减少 conv 层 filter 的数量，或者删掉某些 conv 层，不建议对靠前的 conv 层进行操作，可能会丢掉很多 feature，调整了网络结构后，一定要计算最后的输出是否满足 SxSx(B*5+C)
- 结构中的 1x1 和 3x3 的 conv. 层，启发来至于 GoogLeNet(Inception)。**1x1 卷积的作用后面会把链接补充在这**。
- 最后一层使用线性激活函数（现在还不太明白为什么最后一层用 f(x)=x 的线性激活？），其他层用 leaky RELU 激活函数。

```c
// darknet activations.h 源码中定义的这两个激活函数
static inline float linear_activate(float x){return x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
```

- CNN 基础系统的介绍，参考 [cs231n/convolutional networks](http://cs231n.github.io/convolutional-networks/) 。



### **4. YOLO 损失函数**
----
在理解 YOLO 损失函数之前先思考，目标检测的任务是识别和定位，因此损失函数的设计也是围绕**定位误差**和**识别误差**来展开，YOLO 模型输出的预测值是 bbox 的坐标 (x, y, w, h) 及置信度（confidence）和对应类别 (class)，其中

**(x, y, w, h, confidence) 对应定位预测，更细分的话 （x, y, w, h）对应坐标预测
(class) 对应类别预测**

YOLO 中采用 **sum-squared error** 来计算总的 loss，具体如下：

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo5.png"/></div>





- **part 1**，坐标预测产生的 loss。

  其中 **𝟙 obj** 定义如下：

  **𝟙 obj** = 1，grid cell i 中包含一个 object 的中心，该 grid cell 中第 j 个预测的 bbox 负责这个 object

  **𝟙 obj** = 0，otherwise (grid cell i 不包含 object，或者 包含 object，但没有没有预测的 bbox)

   取较大的权重，**λ noobj** *= 5*（具体要根据数据集），因为有很多 grid cell 中没有 object，在 part3 中会计算 confidence 的 loss，累计起来可能导致，没有 object 的 grid cells 产生的 loss，比有 object 的 grid cells 产生的 loss 对梯度贡献更大，造成网络不稳定发散。

> YOLO 预测每个 grid cell 时可能有多个 bbox。 在训练时，YOLO 只希望一个 bbox 负责 grid cell 对应的 object。 如何只取一个？计算该 grid cell 对应的所有预测的 bbox 与 ground-truth 计算IOU，获得最大 IOU 的那个预测 bbox 成为责任重大的那一个 。为何 w, h 时带根号，文中解释是为了强调 **在 large box 产生小的偏差 比 在 small box 产生的偏差** 影响要小



- **part 2**，**part 3**，confidence 预测产生的 loss。

  其中 **𝟙 obj** 定义如下：

  **𝟙 obj** = 1，grid cell i 对应预测的 bbox 中包含 object

  **𝟙 obj** = 0，otherwise

  **𝟙 noobj**，与之相反，当 bbox 不包含 object 时，预测的 confidence=0

  因为大部分 bbox 可能没有 object，造成 part2, part3 两部分的 loss 不平衡，因此在 part3 中增加一个更小的权重 **λ noobj** *= 0.5*（具体要根据数据集），part2 中则保持正常权重 1


- **part 4**， 类别预测产生的 loss。

  **𝟙 obj** = 1，有 object 中心落在 grid cell i 中

  **𝟙 obj** = 0，otherwise

  注意这部分并没有惩罚 gridcell 中没有 object 的情况



### **5. YOLO 训练**
-----
**这部分后面补充 YOLOv2，YOLOv3 的训练链接在这**，训练 YOLOv1 已经没有太多意义。



### **6. YOLO detection 过程**
----
这部分参考一个很好的 [YOLO ppt](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000#slide=id.p)。

以 S = 7，B =2，C=20 为例，既 grid cell 7x7，每个 grid cell 预测 2 个 bboxes，一共有 20 个 classes.

- 首先直接看 YOLO 模型的输出，如下图，最后输出 7x7x30，grid cell 是 7x7，然后每个 grid cell 要对应 30 个值，前面 10 个 对应 2 个 bboxes 的（x, y, w, h, c），后面 20 个对应 20 个类别的条件概率，将 置信度 与条件概率得到 confidence score。每一个 bbox 对应 20x1 的 confidence score.

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo6.png"/></div>



- **遍历所有的 grid cell** 就可以的到，如下图。

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo7.png"/></div>



- 对预测的 bbox 进行筛选。

  将 score 小于 某个阈值的置0。

  然后根据 score 进行将序排序

  然后用 **NMS** 进一步筛选掉多余的 bbox.

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo8.png"/></div>





### **7. 总结**
----

当时，YOLO 算法与其他 state-of-the-art 的对比

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/yolo9.png"/></div>



当 R-CNN，Fast R-CNN，Faster R-CNN 等算法一统江湖的时候，要想突出重围，不一定要全面压制，例如 YOLO 系列和 SSD 系列等在实时性上实现了压制，虽然精度上略逊一筹，同样得到了众多目标检测研究者或工作者的青睐。此外，能够看到一个系列的算法一步步优化，很 nice，例如 R-CNN 发展到了 Faster R-CNN，Mask R-CNN等，YOLO 到了 YOLO v3，最重要的还是学习算法优化的思想，多思考。

后面想补充的博客：

1. R-CNN 系列
2. SSD 系列，算法原理及实现
3. YOLOv2, YOLOv3，算法原理以及实现
4. 深度学习框架



### **links**
----
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [darknet](https://pjreddie.com/darknet/)
- [darknet github](https://github.com/pjreddie/darknet)
- [SSD: Single Shot Multibox Detector](https://arxiv.org/pdf/1512.02325.pdf)
- [YOLO ppt](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000#slide=id.p)
- [A nice YOLO Blog](https://hackernoon.com/understanding-yolo-f5a74bbc7967)




