# Backpropagation

compute gradient -- **computational graph**

backpropagation procedure:

- forward pass: compute outputs
- backward pass: compute derivatives $\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}...$  
  - use "chain rule": $\frac{\partial f}{\partial y}=\frac{\partial q}{\partial y}\frac{\partial f}{\partial q}$
    - $\frac{\partial f}{\partial y}$: Downstream Gradient
    - $\frac{\partial q}{\partial y}$: Local Gradient
    - $\frac{\partial f}{\partial q}$: Upstream Gradient

​	每个node：

![image-20250429130646516](D:\adolph\mark\image-20250429130646516.png)

![image-20250429131734563](D:\adolph\mark\image-20250429131734563.png)

有时候用graph会更麻烦，可以合并node，比如sogmoid，如下

![image-20250429132127167](D:\adolph\mark\image-20250429132127167.png)

![image-20250429132543173](D:\adolph\mark\image-20250429132543173.png)



vector case

![image-20250429133918383](D:\adolph\mark\image-20250429133918383.png)



对于ReLU函数，Jacobi矩阵中对角线上以外的元素全是0，其他深度学习中用到的函数他们的 Jacobi矩阵也通常很稀疏(sparse)，存储矩阵完整内容很浪费，所以一般use **implicit** multiplication

![image-20250429134652405](D:\adolph\mark\image-20250429134652405.png)





**Backpropagation with Matrices**

![image-20250503003226697](D:\adolph\mark\image-20250503003226697.png)

由图可知，完整表示Jacobi矩阵需要超大内存，必须implicitly表示

可以element-wise计算local gradient slice  $\frac{dy}{dx_{i,j}}$

$\frac{dL}{dx_{i.j}}=\frac{dy}{dx_{i,j}}\cdot \frac{dL}{dy}=w_{j,:}\cdot\frac{dL}{dy_{i,:}}$	可以自己写一写推一下

![image-20250503005054604](D:\adolph\mark\image-20250503005054604.png)

右边是implicitly表示的Jacobi矩阵



**Forward-Mode Automatic Differentiation**