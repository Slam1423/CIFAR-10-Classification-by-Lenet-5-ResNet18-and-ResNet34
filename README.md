# CIFAR-10-Classification-by-Lenet-5-and-ResNet18

采用Lenet5和ResNet18两种卷积神经网络结构对CIFAR10数据集训练分类器，最终表现是ResNet18明显好于Lenet5，从而印证了残差网络的强大的训练能力。实验步骤如下：

1、数据的处理与准备

我们使用python的torch模块中自带的CIFAR10数据集，将CIFAR10数据集中的图片大小统一设为32*32，并且转换为张量数据类型。

2、模型构建、训练以及结果

（1）Lenet5

这里Lenet5由两层卷积层、两层average pooling层以及三层全连接层组成，结构如下：

![image](https://github.com/Slam1423/CIFAR-10-Classification-by-Lenet-5-and-ResNet18/blob/main/lenet5.jpg)

由于CIFAR10中图片属于10个类别，因此我们最后一层全连接层的输出维度为10，并且采用交叉熵为损失函数，采用Adam作为优化器，batch_size=32，然后进行了40次迭代，最终分类准确率维持在54%左右。我们认为其准确率无法进一步提升的原因是Lenet5本身层次较浅，无法学习到图片中深层次的特征，因而分类准确度很难再有大的突破。

（2）ResNet18

为弥补Lenet5深度不够的缺陷，我们采用更深层次的残差网络——ResNet18，我们构建的ResNet18结构大体由卷积层、Batch Normalization层、residual unit1、residual unit2、residual unit3、residual unit4以及全连接层构成。其中，每个residual unit都由两个卷积层、两个Batch Normalization层构成。

我们同样将最后一层全连接层的输出维度设置为10，以匹配预测类别数。采用交叉熵作为损失函数，Adam作为优化器，batch_size=32，然后运行了6个小时，共7次迭代（本人的笔记本电脑不含cuda，只能使用cpu训练，速度较慢），但每一次迭代后分类准确度都有明显提高，最终测试集上准确率达到了78.42%，相较于Lenet5网络结构，ResNet18训练效果提升了很多。下图展现了两种模型在测试集上分类准确度随时间的变化情况。

![image](https://github.com/Slam1423/CIFAR-10-Classification-by-Lenet-5-and-ResNet18/blob/main/Lenet5%E4%B8%8EResNet18%E5%AF%B9%E6%AF%94.png)

由此可以发现，Lenet5网络结构准确率达到54%左右之后便无法继续提升，并且还会出现一定的下降，主要是其深度不够造成的。而ResNet18网络则弥补了这一缺陷，在实验中的7次迭代中，每一次迭代后分类准确率都有明显的提高，可以预见的是，如果我们在第7次迭代之后继续很多步迭代的话，它可能达到一个特别高的准确率。这主要得益于它的深度网络结构能够对图片的深层次特征进行提取和挖掘。

下一步工作的展望：

1、时间充裕的情况下，使用ResNet18网络再进行更多步的迭代，直到准确度的提升达到饱和为止。 

2、尝试更深层的残差网络模型，如ResNet34、ResNet50等，应该会有更大的提升。
