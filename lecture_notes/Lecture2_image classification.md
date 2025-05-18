# Image Classification

image classification这个任务不像sort a set of numbers一样你可以用明确的算法、步骤去完成，且针对不同的set都可以用同一种算法完成，如果你要人为地去提取图片特征，比如提取猫的照片中线条轮廓，那样会非常难且得到的模型只适用于对猫分类，如果任务换成同类型但不同对象比如对星系分类的话则需要从头开始重新提取对象特征。一个有用的方法是使用机器学习

machine learning: **data-driven-approach**	instead of写一个具体的算法函数，而是让机器去学习数据集，自己得出模型

`tain(images, labels), predict(model, test_images)`

1. collect a dataset of images and labels
2. use machine learning to train a classifier
3. evaluate the classifier on new images



**K-Nearest Neighbors**

> - During training, the classifier simply memorizes the training data
> - During testing, test images are compared to each training image; the predicted label is the majority vote among the K nearest training examples.

with the right choice of distance metric, we can apply K-Nearest Neighbor to any type of data

distance metric to compare images:

L1 distance (Manhattan曼哈顿距离): $d_1(I_1, I_2)=\sum_p|I_1^p-I_2^p|$		pixel-wise absolute value differences

L2 distance (Euclidean 欧几里得)

如果数据足够多，KNN几乎可以拟合任何函数

随着数据维度增加，拟合所需数据点呈指数级增长（BAD）

`xrange()`在Python3中被删除，Python2中用法跟range类似，但返回迭代器而不是列表。Python3中range也返回迭代器

>  we can afford slow training, but we need fast testing!

数据集训练方法选择：

1. 用整个数据集训练 BAD!!! -> 会过拟合（K=1 always works perfectly）不知道在其他数据上的效果，泛化能力差
2. 分成训练集和测试集 BAD!!! -> 使用测试集来修改超参数，会让测试集信息不再不可见，会泄露到模型中，无法公平评估模型效果
3. 分成训练集、验证集、测试集（训练后根据验证集调整超参数，最后用测试集评估效果  BETTER!!!
4. Cross-Validation   BEST  (but expensive)![image-20250426001309965](D:\adolph\mark\image-20250426001309965.png)

KNN on raw pixels is **seldom used** because: (1) very slow at test time (2) distance metrics on pixels are not informative