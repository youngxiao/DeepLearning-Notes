
#  Why？ 1x1 卷积


### **1. 前言**
------

往往很多人喜欢把事情搞得复杂，显得高大上，却并不实用，殊不知有时候简单的想法才是最美的。就好像是一堆巧妙的数学公式推导，常常有某一个极简的数学公式在作支撑，奇技淫巧无处不在。此外，研究算法的乐趣之一就是存在这些神奇的地方。如果是自己发现会很有成就感，如果在别人的思想中发现也会产生共鸣。



今天要记录 CNN 中的 **1x1 卷积**的神奇之处，或者说有什么用？当然我承认，我也是之前被问到这个问题！！当时想法是因为 input 的 feature maps 是有 **channel** 的，通过 1x1 卷积后，就将这些 channel 的特征线性组合起来，使得不同 channel 的特征联系起来，特征更丰富等等。也不能说当时的想法错误，只能说思考的不够全面。专门花了点时间去思考这个问题，当然从论文和前人的博客中也学到了很多，最后会添加部分 links.



### **2. 来历**
-----
1x1 卷积不管是不是最早出现在 [Network In Network](https://arxiv.org/abs/1312.4400) (NIN)，这都是一篇很重要的文章，两点贡献：

- **mlpconv, 即引入 1x1 卷积**。

  传统的 CNN 的一层卷积层相当于一个线性操作，如下图 a，所以只提取了线性特征，隐含了假设特征是线性可分的，实际却并非如此，NIN 中引入 下图 b 中的 mlpconv layer，实质是**像素级的全连接层**，等价于 1x1 卷积，在其后跟 ReLU激活函数，引入更多的非线性元素。

- 将 CNN 中的**全连接层**，采用 **global average pooling** 层代替。

  全连接层容易过拟合，减弱了网络的泛化能力。



<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/1conv1.png"/></div>



在这之后，另一个经典的网络 [GoogLeNet](https://arxiv.org/pdf/1409.4842v1.pdf) 的 **Inception** 结构中沿用了 NIN 中的 1x1 操作，如下图。是金子总会发光，然后是被应用到各个网络中去。



<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/1conv3.png"/></div>



### **3. 1x1 卷积作用**
-----



- **降维和升维**

  简单的解释**降维**，例如，input 的 feature maps 是 16x16，channel=32，通过 一个 1x1，filters 数量为 8 卷积层，output 为 16x16x8。更深入的解释，以下图为例，如下图 a 为最原始的卷积，下图 b 中改进，引入 1x1 卷积，目的是：

  - 在网络中增加 1x1 卷积，**使得网络更深**
  - 网络的深度增加并没有增加权重参数的负担，反而大大减少，因为 feature maps 在输入到 3x3 或者 5x5 卷积时，**先经过 1x1，已经降低了维度。模型效果并没有降低**？（没有，这跟 1x1 卷积的后面几点作用有关）。如下例子，**参数数量大大减少**，随着网络深度加深，会成倍减少。
  ```c
  // 假设
  input feature maps:28×28  channel:256
  1x1 convolutional layer:1x1 channel:16
  5x5 convolutional layer:5x5 channel:32
      
  // 那么
  // 1.在下图 a 中
  卷积核参数量 = 28x28x256x5x5x32 = 160.5M
  
  // 2.在下图 b 中
  5x5 卷层之前经过了 1x1 卷积
  输入的 size 为 28x28x256 经过 1x1 卷积后 size 为 28x28x16
  卷积核参数量 = 28x28x256x1x1x16 + 28x28x16x5x5x32 = 13.25M
  ```

<div align=center><img src="https://github.com/youngxiao/DeepLearning-Notes/raw/master/pic/1conv2.png"/></div>





再来看看**升维**，通常是在一个卷积之后添加一个 1x1 卷积来升维。例如经过一个卷积后，输出 feature maps size 为 16x16x32，作为 1x1x128 卷积的输入，则输出 16x16x128 的 feature maps 实现升维。



- **跨通道信息交互（cross-channel correlations and spatial correlations）**

  1x1卷积核，从图像处理的角度，乍一看也没有意义，在网络中，这样的降维和升维的操作其实是 channel 间信息的线性组合变化。

  另外补充，cross-channel correlation 和 spatial correlation的学习可以进行解耦。1x1的卷积相当于学习了feature maps之间的cross-channel correlation。实验证明了这种解耦可以在不损害模型表达能力的情况下大大减少参数数量和计算量。但是需要注意的是，1x1 的卷积层后面加上一个 normal 的卷积层，这种解耦合并不彻底，正常卷积层仍然存在对部分的 cross-channel correlation 的学习。之后就有了 **depth-wise seperable convolution**(后面记录 MobileNet 后，在这添加链接)。在 depth-wise seperable convolution中，1x1 的卷积层是**将每一个 channel 分为一组**，那么就不存在对cross-channel correlation的学习了，就实现了对cross-channel correlation和spatial correlation的彻底解耦合。这种完全解耦的方式虽然可以大大降低参数数量和计算量，但是正如在 **mobile net** 中所看到的，性能会受到很大的损失。


- **增加非线性特性**

  1x1卷积核，可以在保持 feature maps  size不变的（即不损失分辨率）的前提下大幅增加非线性特性（利用后接的非线性激活函数）。



### **4. 总结**
----
1x1 卷积在图像处理的角度，乍一看好像没什么意义，但在 CNN 网络中，能实现降维，减少 weights 参数数量，能够实现升维，来拓宽 feature maps，在不改变 feature maps 的 size 的前提下，实现各通道之间的线性组合，实际上是通道像素之间的线性组合，后接非线性的激活函数，增加更多样的非线性特征。这就是为什么 GoogLeNet 用 1x1 卷积来降维，减少了计算量，但模型效果却没有降低，此外网络深度更深。可以说 1x1 卷积很 nice.



### **links**
----
- [Network In Network](https://arxiv.org/abs/1312.4400)
- [GoogLeNet](https://arxiv.org/pdf/1409.4842v1.pdf)
- [A NIN blog](https://www.cnblogs.com/makefile/p/nin.html)
- [One by one convolution](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)
- [1x1 conv.](https://zhuanlan.zhihu.com/p/35814486)



### **others**
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com


