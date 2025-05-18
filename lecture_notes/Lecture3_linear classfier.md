# Linear Classifier

$f(x, W)$

不同角度理解：

algebra viewpoint:

线性 -> f(cx, W) = c * f(x, W)

bias项可以加到权重矩阵最后一列，然后在数据向量最后加一个1元素

visual viewpoint:

将权重矩阵每行单拉出来reshape成与输入图像对应的维度，对应元素相乘加和

从这个角度看，每个种类像是一个template，分类时相当于往模板上套



Loss function

- multiclass SVM class 
  - Hinge loss:  $L_i=\sum_{j\neq y_i}max(0, s_j-s_{y_i}+1)$		($y_i$ is label)
  - if scores are small random values -> loss = C - 1   (C is the number of labels)
- Cross-Entropy Loss (Multinomial Logistic Regression)	wants to interpret raw classifier score as probabilities
  - $P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_j e^{s_j}}$	softmax function
  - $L_i=-logP(Y=y_i|X=x_i)$
  - if scores are small random values -> loss = -log(C)



regularization	(purpose?)

​	$L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i, W),y_i)+\lambda R(W)$

​	L2 regularization  $\sum_k \sum_l W_{k,l}^2$  likes to "spread out" the weights