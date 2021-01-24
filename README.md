# CIFAR-10-Classification-by-Lenet-5,ResNet18 and ResNet34

采用Lenet5、ResNet18以及ResNet34三种卷积神经网络结构对CIFAR10数据集训练分类器，最终表现是ResNet18和ResNet34分类效果明显好于Lenet5，但训练时间更长。而ResNet18和ResNet34分类效果相近。（在DBCloud 1080Ti上训练）具体实验细节如下：

1、数据的处理与准备

我们使用python的torch模块中自带的CIFAR10数据集，将CIFAR10数据集中的图片大小统一设为32*32，并且转换为张量数据类型。

2、模型构建、训练以及结果

（1）Lenet5

这里Lenet5由两层卷积层、两层average pooling层以及三层全连接层组成，结构如下：

![image](https://github.com/Slam1423/CIFAR-10-Classification-by-Lenet-5-and-ResNet18/blob/main/lenet5.jpg)

由于CIFAR10中图片属于10个类别，因此我们最后一层全连接层的输出维度为10，并且采用交叉熵为损失函数，采用Adam作为优化器，batch_size=32，然后进行了40次迭代，最终分类准确率维持在54%左右。我们认为其准确率无法进一步提升的原因是Lenet5本身层次较浅，无法学习到图片中深层次的特征，因而分类准确度很难再有大的突破。

（2）ResNet18

为弥补Lenet5深度不够的缺陷，我们采用更深层次的残差网络——ResNet18，我们构建的ResNet18结构大体由卷积层、Batch Normalization层、residual unit1、residual unit2、residual unit3、residual unit4以及全连接层构成。其中，每个residual unit都由两个卷积层、两个Batch Normalization层构成。参数设置：epoch=200，batch_size=32。

![image](https://github.com/Slam1423/CIFAR-10-Classification-by-Lenet-5-and-ResNet18/blob/main/Lenet5%E4%B8%8EResNet18%E5%AF%B9%E6%AF%94.png)

由此可以发现，Lenet5网络结构准确率达到54%左右之后便无法继续提升，并且还会出现一定的下降，其原因应为其深度不够。而ResNet18和ResNet34准确率有更为持久的提升，但在训练500s之后分类准确率提升效果不再明显，ResNet18分类效果稍好于ResNet34的分类效果。

下一步工作的展望：

1、阅读http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130 上具有更高准确率的论文，优化网络结构。

2、尝试更深层的残差网络模型，如ResNet50、ResNet101等，若分类效果与ResNet18和ResNet34相近，则分析其分类效果无法进一步改善的原因。
