# Optimization

$w^∗=arg\ min_w\ L(w)$	**arg minw L(w)** 的意思是“找到能使 L(w) 取得最小值的 w”



method to compute gradient:

- numeric gradient

​	slow: O(#dimensions)

​	approximate 不精确

- analytic gradient	(如何实现？)

​	exact, fast, but error-prone

in practice: always use analytic gradient, but check implementation with numerical gradient.   --"gradient check"



**Gradient Descent**

```python
w = initialize_weights()
for t in range(num_steps):
    dw = compute_gradient(loss_fn, data, w)
    w -= learning_rate * dw
"""
hyperparameters:
- weight initialization method
- number of steps
- learning rate
"""
```

上面方法计算full sum is expensive

改进如下：

```python
# Stochastic gradient descent (SGD) 随机梯度下降法
# think of loss as an expectation over the full data distribution E[L(x, y, W)]
w = initialize_weights()
for t in range(num_steps):
    mini_batch = sample_data(data, batch_size)	# batch_size常用取值:32/64/128
    dw = compute_gradient(loss_fn, minibatch, w)
    w -= learning_rate * dw
"""
hyperparameters:
- weight initialization method
- number of steps
- learning rate
- batch size
- data sampling
"""
```



![image-20250428200653382](D:\adolph\mark\image-20250428200653382.png)

![image-20250428201133801](D:\adolph\mark\image-20250428201133801.png)

to overcome these problems in SGD: 

- **SGD + Momentum**

![image-20250428201303934](D:\adolph\mark\image-20250428201303934.png)

![image-20250428201649830](D:\adolph\mark\image-20250428201649830.png)

![image-20250428202217988](D:\adolph\mark\image-20250428202217988.png)

Nesterov Momentum:

$v_{t+1}=\rho v_t-\alpha \nabla f(x_t+\rho v_t)$

$x_{t+1}=x_t+v_{t+1}$

- **AdaGrad**

```python
grad_squared = 0
for t in range(num_steps):
    dw = compute_gradient(w)
    grad_squared += dw * dw
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

更新参数时，用原始学习率 `learning_rate` 除以对应维度的累积平方根 `sqrt(grad_squared)`（再加上一个 `1e-7` 的小常数以防除零），实现对每个参数的自适应缩放

优势：

\- 对于更新频繁（梯度大）的参数，累积平方和 GtG_tGt 会迅速增大，导致分母变大，进而使得学习率自动变小；

\- 对于不常更新（梯度小或稀疏）的参数，学习率相对较大，有助于加快收敛。

\- 因此，AdaGrad 能够根据每个参数的历史梯度信息，动态地分配“**每参数学习率**”，常称为“**自适应学习率**”（adaptive learning rates）。

- RMSProp: "Leak Adagrad" 对Adagrad进行改进（改进内容问GPT

![image-20250428203904332](D:\adolph\mark\image-20250428203904332.png)

- Adam (**very common in practice!**) : RMSProp + Momentum



![image-20250428204506802](D:\adolph\mark\image-20250428204506802.png)

Adam is a good default choice in many cases

SGD+Momentum can outperform Adam but may require more tuning



first-order optimization: make linear approximation

second-order optimization: make quadratic approximation   use Hessian matrix -> impractical 复杂度太高