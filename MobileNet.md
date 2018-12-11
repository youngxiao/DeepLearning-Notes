
#  为何 MobileNet 及其变种（如 ShuffleNet）速度很快

> 这篇文章是从 [Medium](https://medium.com/) 上看到的一篇 blog，讲的挺好的，直接翻译过来了。

翻文地址：[Why MobileNet and Its Variants (e.g. ShuffleNet) Are Fast](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)





### **1. 介绍**

------

在本文中，概述了高效的 CNN 模型（如 MobileNet 及其变体）中使用的构建块，并解释了它们如此高效的原因。 特别地，提供了关于如何完成 **空间（spatial）和通道（channel）**域中卷积的直观说明。



### **2. 高效模型中使用的构建块**
-----


在解释特定的高效的 CNN 模型之前，让我们检查 CNN 模型中使用的构建块的计算成本，并了解如何在空间和信道域中执行卷积。



<div align=center><img height="250" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet1.png"/></div>



设 **HxW** 表示输入 feature map 的空间大小，**N** 表示输入 channel 的数量，**KxK** 表示卷积核的大小，**M** 表示输出 channel 的数量，标准卷积的计算成本为 **HxWxNxKxKxM（记为HWNK²M）**。

这里重要的一点是标准卷积的计算成本与（1）feature map 空间 HxW，（2）卷积核 KxK，（3）输入和输出通道的数量 NxM 成比例。

当在 spatial 和 channel 上执行卷积时，需要上述的计算成本。 下文中的描述，可以通过分解该卷积来加速CNN。



#### 卷积（Convolution）

上述中，我提供了一个直观的例子，说明如何在标准卷积中完成 spatial 和 channel 域上卷积，其计算成本为**HWNK²M**。

我在输入和输出之间连线以显示输入和输出之间的依赖关系。 **连线的条数**大致分别表示 spatial 和 channel 域中卷积的计算成本。

<div align=center><img height="130" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet2.png"/></div>



例如，最常用的卷积 conv3x3 可以如上所示进行可视化。 我们可以看到输入和输出在 spatial 域中局部连接，而在通道域中，它们是全连接的。



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet3.png"/></div>



接下来，上图显示 [conv1x1 [1]](https://arxiv.org/abs/1312.4400)，用于改变 channel 的大小。 该卷积的计算成本是 **HWNM**，因为内核的大小是1x1，与 conv3x3 相比，计算成本降低到 1/9。 该卷积用于跨通道的信息交互，具体请看 [1x1 卷积的作用](https://blog.csdn.net/u010986080/article/details/84945170)。





#### 分组卷积（Grouped Convolution）

分组卷积是卷积的变体，其中**输入 feature map 的 channel 被分组，并且对于每个分组的 channel 独立地执行卷积。**

假设 **G** 表示组数，分组卷积的计算成本是 **HWNK²M/ G**，与标准卷积相比，计算成本降低到 **1 / G**.



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet4.png"/></div>

转换为 conv3x3 且G = 2 的情况。 我们可以看到，channel 域中的连接数变得小于标准卷积，这表明计算成本较低。



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet5.png"/></div>

将 conv3x3 分组并且 G = 3 的情况。 连接变得更加稀疏。




<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet6.png"/></div>
将 conv1x1 分组且 G = 2 的情况。 因此，conv1x1也可以分组。 在 [**ShuffleNet**]() 中使用这种类型的卷积。




<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet7.png"/></div>
将 conv1x1 分组，G = 3 的情况。



#### Depthwise Convolution

在 [Depthwise  Convolution [4]](https://arxiv.org/abs/1610.02357) 中，对每个输入 channel 独立地执行卷积。 它也可以定义为分组卷积的特殊情况，其中输入和输出 channel 的数量相同，G 等于 channel 数。



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet8.png"/></div>
如上图， depthwise conv 显著的降低了 channel 域中的计算成本。



#### Channel Shuffle

Channel shuffle 是一个改变了 ShuffleNet [5] 中使用 channel 的顺序。 该操作由 **tensor reshape** 和 **transpose** 实现。



更确切地说，让 GN'（= N）表示输入 channel 的数量，输入 channel 维度首先被 reshape 为（G，N'），表示 G 组，然后将（G，N'）转置为（N'，G），最后 flatten into 与输入相同的 size。 这里，G表示用于 grouped convolution 的组数，其与 channel shuffle layer 一起被应用到 ShuffleNet 中。

虽然不能根据乘和加来算 channel shuffle 的计算成本，但应该存在一些开销。



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet9.png"/></div>
channel shuffle G = 2 的情况。 不执行卷积，简单地改变 channel 的顺序。



<div align=center><img height="61" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet10.png"/></div>
channel shuffle，G = 3。





### **3. 高效的模型**
-----
在下文中，对于高效的 CNN 模型，直观的说明了它们为什么高效，以及如何在 spatial 和 channel 域中完成卷积。

#### ResNet（Bottleneck 版本）

[ResNet [6]](https://arxiv.org/abs/1512.03385) 中使用了具有 **bottleneck** 的 **残差单元（Residual unit）**，是进一步与其他模型进行比较的良好起点。



<div align=center><img height="210" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet11.png"/></div>

如上图，具有 bottleneck 的残差单元由 conv1x1，conv3x3 和 conv1x1 组成。 第一个 conv1x1 减小了输入 channel 的 size，既降维，使得减小了后续相对昂贵的 conv3x3 的计算成本。 最终的 conv1x1 恢复输出 channel 的维度。



#### ResNeXt

[ResNeXt [7]](https://arxiv.org/abs/1611.05431) 是一种高效的 CNN 模型，可以看作是 ResNet 的特例，其 conv3x3 被分组的 conv3x3 取代。 通过使用高效的分组转换，与 ResNet 相比，conv1x1 中的 channel 降低率变得适中，从而以相同的计算成本获得更好的准确性。

<div align=center><img height="210" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet12.png"/></div>





#### MobileNet（可分离的 Conv）

[MobileNet [8]](https://arxiv.org/abs/1704.04861) 中应用可分离卷积模块（separable convolution modules），由 **deepwise conv** 和 conv1x1（**pointwise conv**）组成。

<div align=center><img height="134" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet13.png"/></div>
separable conv 在 spatial 和 channel 域中独立地执行卷积。 这种卷积的计算成本从 **HWNK²M** 降到了 **HWNK² (depthwise) + HWNM (conv1x1)**，总共 **HWN(K²+ M)** 的计算成本。 通常，M >> K 2（例如K = 3且M≥32），大致降低到 **1/8 - 1/9**。

这里重要的一点是计算成本的瓶颈现在是 conv1x1！



#### ShuffleNet

ShuffleNet 的动机是 conv1x1 是如上所述的 separable conv 的瓶颈。 虽然 conv1x1 已经很高效，并且似乎没有改进的余地，但是conv1x1可以用于此目的！

<div align=center><img height="280" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet14.png"/></div>
上图说明了 ShuffleNet 的模块。 这里重要的构建块是 **channel shuffle layer**，它在分组卷积中“混合”组间 channel 的顺序。 在没有 channel shuffle 的情况下，分组卷积的输出从不在组之间被利用，导致准确性降低。



#### MobileNet-V2

[MobileNet-v2 [9]](https://arxiv.org/abs/1801.04381) 采用了类似于 ResNet中的 **残差单元** 结构， 修改版本的 残差单元，其中 conv3x3 由 depthwise conv 代替。

<div align=center><img height="210" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet15.png"/></div>
从上图可以看出，与标准 bottleneck 结构相反，第一个 conv1x1 增加了channel 维度，然后执行了 depthwise conv，最后一个 conv1x1 降低了 channel 维度。



<div align=center><img height="210" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet16.png"/></div>
通过对上述结构重新排序，并将其与 MobileNet-v1（separable conv）进行比较，我们可以看到该结构如何工作（此重新排序不会改变整体模型架构，MobileNet-v2 是正是用的此模块）。

也就是说，上述模块被视为 separble conv 的修改版本，其中 separble conv 中的单个 conv1x1 被分解为两个 conv1x1。假设 **T** 表示 channel 维数的扩展因子，两个 conv1x1 的计算成本是 **2HWN²/ T**，而 separable conv 中 conv1x1 的计算成本是 **HWN²**。在 ShuffleNet[5] 中，使用 T = 6，将 conv1x1 的计算成本降低了3倍（通常为T / 2）。



#### FD-MobileNet

最后，介绍 [Fast-Downsampling MobileNet（FD-MobileNet）[10]](https://arxiv.org/abs/1802.03750)。在此模型中，与 MobileNet 相比，在较早的层中执行下采样。这个简单的技巧可以降低总计算成本。原因在于传统的下采样策略和 separable conv。

从 VGGNet 开始，许多模型采用相同的下采样策略：执行下采样，然后将后续层的 channel 维数加倍。 **对于标准卷积，计算成本在下采样后不会改变，因为它由HWNK²M定义**。 然而，对于 separable conv，其下采样后的计算成本变小; 它从 **HWN(K² + M) 减小到 (H/2  W/2  2N(K² + 2M) = HWN(K²/2 + M)**。 当 M 不那么大（即较早的层）时，减小就是相对显性的。

最后用一下 cheat shhet 结束这篇文章。

<div align=center><img height="720" src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/mobilenet17.png"/></div>



### 参考文献
------
[[1] M. Lin, Q. Chen, and S. Yan, “Network in Network,” in Proc. of ICLR, 2014.](https://arxiv.org/abs/1312.4400)

[[4] F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” in Proc. of CVPR, 2017.](https://arxiv.org/abs/1610.02357)

[[5]  X. Zhang, X. Zhou, M. Lin, and J. Sun, “ShuffleNet: An Extremely  Efficient Convolutional Neural Network for Mobile Devices,” in  arXiv:1707.01083, 2017.](https://arxiv.org/abs/1707.01083)

[[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. of CVPR, 2016.](https://arxiv.org/abs/1512.03385)

[[8] A. G.  Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M.  Andreetto, and H. Adam, “Mobilenets: Efficient Convolutional Neural  Networks for Mobile Vision Applications,” in arXiv:1704.04861, 2017.](https://arxiv.org/abs/1704.04861)

[[9]  M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen, “MobileNetV2:  Inverted Residuals and Linear Bottlenecks,” in arXiv:1801.04381v3,  2018.](https://arxiv.org/abs/1801.04381)

[[7]  S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated Residual  Transformations for Deep Neural Networks,” in Proc. of CVPR, 2017.](https://arxiv.org/abs/1611.05431)

[[10] Z. Qin, Z. Zhang, X. Chen, and Y. Peng, “FD-MobileNet: Improved MobileNet with a Fast Downsampling Strategy,” in arXiv:1802.03750, 2018.](https://arxiv.org/abs/1802.03750)


### **others**
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com
* conv1x1: [1x1 conv 的作用](https://blog.csdn.net/u010986080/article/details/84945170)


