**construction**

`a = torch.tensor([[1,2,3], [2,3,4]])`

`type(a)`

`a.dim()`返回a的rank  **这里的rank跟线性代数里的rank不一样！！这里是指tensor的维度数，比如二维张量rank为2

`a.shape` 返回形状	torch.Size([2, 3])

`.item()` 将pytorch到标量转化为Python中的变量类型	`a[0, 1].item()`

`torch.zeros(张量尺寸)`

`torch.ones(张量尺寸)`

`torch.rand(张量尺寸)`

`y = torch.zeros_like(x)  ` # Create an empty matrix with the same shape as x

`torch.eye(n, m=None, dtype=None, device=None)`: 创建单位矩阵，n为行数，m为列数

`a += 1` a的所有元素都+1



**datatypes**

如果用来生成tensor的列表里有整数和浮点数，最后生成的元素都是浮点数

`torch.tensor([...], dtype=...)`	dtype可选：`torch.float32, torch.bool, torch.int64, ...`

ones, zeros等也可用dtype参数

`.to()`: cast a tensor to another datatype

```python
x0 = torch.eye(3, dtype=torch.int64)
x1 = x0.float()  # Cast to 32-bit float
x2 = x0.double() # Cast to 64-bit float
x3 = x0.to(torch.float32) # Alternate way to cast to 32-bit float
x4 = x0.to(torch.float64) # Alternate way to cast to 64-bit float
```

`torch.zeros_like()`: create new tensors with the same shape and type as a given tensor

Tensor objects have instance methods such as `.new_zeros()` that create tensors the same type but possibly different shapes

The tensor instance method `.to()`can take a tensor as an argument, in which case it casts to the datatype of the argument.

```python
x0 = torch.eye(3, dtype=torch.float64)  # Shape (3, 3), dtype torch.float64
x1 = torch.zeros_like(x0)               # Shape (3, 3), dtype torch.float64
x2 = x0.new_zeros(4, 5)                 # Shape (4, 5), dtype torch.float64
x3 = torch.ones(6, 7).to(x0)            # Shape (6, 7), dtype torch.float64)
```



**index**

slice syntax: `start:stop` or start:stop:step

negtive index: 从最后一个往回数，最后一个是-1

中括号里用逗号是提取单个元素，冒号是区间切片，用逗号但某个维度加冒号是提取该维度某个区间

```python
a = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[1, :])	# Get row 1, and all columns.
print(a[1])  # Gives the same result; we can omit : for trailing dimensions
print(a[:, 1])	#Single column
print(a[:2, -3:])	# Get the first two rows and the last three columns
print(a[::2, 1:3])	# Get every other row, and columns at index 1 and 2	注意这里2是step
```

**There are two common ways to access a single row or column of a tensor: using an integer will reduce the rank by one, and using a length-one slice will keep the same rank.

```python
a = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
'''results:
tensor([5, 6, 7, 8]) torch.Size([4])
tensor([[5, 6, 7, 8]]) torch.Size([1, 4])
'''
```

slice切片得到的tensor是原tensor的“引用”，修改它会改变原tensor。解决办法是使用`clone()`

```python
c = a[0, 1:].clone()
```

用slice修改tensor内容：可以用一个数或对应形状的tensor

**注意：Python中函数参数是通过“引用传递”而不是“值传递”！因此函数内部对参数的修改会影响原始参数！**

> When you index into torch tensor using slicing, the resulting tensor view will always be a subarray of the original tensor. This is powerful, but can be restrictive.

We can also use **index arrays** to index tensors; this lets us construct new tensors with a lot more flexibility than using slices.

用数组做索引时，结果是创建新tensor而不是原tensor的引用

```python
a = torch.tensor([[1, 2, 3, 4], 
                  [5, 6, 7, 8], 
                  [9, 10, 11, 12]])
idx = [0, 0, 2, 1, 1]
print(a[idx])	#前两行是a的第一行，第三行是a的第三行，第四五行是a的第二行
print(a[:, idx])#同理，以列为index
#行列都用列表idx0, idx1索引时，结果是分别取a[idx0[i], idx1[i]]合成到一个列表里生成tensor，注意形状
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([0, 1, 2])
print(x[rows, cols])	#取a对角线的元素，结果为tensor([ 1,  6, 11])
```

疑问：用列表和torch.tensor来index tensors有什么区别？

one useful trick:

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# Take on element from each row of a:
# from row 0, take element 1;
# from row 1, take element 2;
# from row 2, take element 1;
# from row 3, take element 0
idx0 = torch.arange(a.shape[0])  # Quick way to build [0, 1, 2, 3]
idx1 = torch.tensor([1, 2, 1, 0])
a[idx0, idx1] = 0
```



**Boolean tensor indexing**

> Boolean tensor indexing lets you pick out arbitrary elements of a tensor according to a boolean mask. Frequently this type of indexing is used to select or modify the elements of a tensor that satisfy some condition.

```python
a = torch.tensor([[1,2], [3, 4], [5, 6]])
print('Original tensor:')
print(a)

# Find the elements of a that are bigger than 3. The mask has the same shape as
# a, where each element of mask tells whether the corresponding element of a
# is greater than three.
mask = (a > 3)
print('\nMask tensor:')
print(mask)

# We can use the mask to construct a rank-1 tensor containing the elements of a
# that are selected by the mask
print('\nSelecting elements with the mask:')
print(a[mask])

# We can also use boolean masks to modify tensors; for example this sets all
# elements <= 3 to zero:
a[a <= 3] = 0
print('\nAfter modifying with a mask:')
print(a)

'''#results:
Original tensor:
tensor([[1, 2],
        [3, 4],
        [5, 6]])

Mask tensor:
tensor([[False, False],
        [False,  True],
        [ True,  True]])

Selecting elements with the mask:
tensor([4, 5, 6])

After modifying with a mask:
tensor([[0, 0],
        [0, 4],
        [5, 6]])
'''
```



**Reshaping operations**

- `.view()`	返回原tensor的引用

```python
x0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
x1 = x0.view(8)	#tensor([1, 2, 3, 4, 5, 6, 7, 8])
x2 = x1.view(1, 8)	#tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
x3 = x1.view(2, 2, 2)	#tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

可以直接用-1来自动取足够的size

```python
x0.view(-1)	#tensor([1, 2, 3, 4, 5, 6, 7, 8])
x0.view(1, -1)	#tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
```

.view()无法实现转置

- `.t()`	It is available both as <u>a function in the torch module</u> and as <u>a tensor instance method</u>

`x.t()` or `torch.t(x)`

- `.transpose()`

> For tensors with more than two dimensions, we can use the function `torch.transpose` to swap arbitrary dimensions.

````python
x1 = x0.transpose(1, 2) # Swap axes 1 and 2
````

- `.permute()`	第n个参数v：make the old dimension v appear at dimension n

- 用.view() 有些情况会发生错误 ([This blog post by Edward Yang](https://www.google.com/url?q=http%3A%2F%2Fblog.ezyang.com%2F2019%2F05%2Fpytorch-internals%2F) gives a clear explanation of the problem.)
  解决办法是先`.contiguous()`再用`.reshape()`



**Tensor operations**

- elementwise operation
  `x + y`, `torch.add(x, y)`, `x,add(y)`效果一样（），sub, mul. div, pow同理

- reduction operation    跟上面一样function in the torch module 和 instance methodson tensor object都可以

  mean, min, max...
  `.sum()`可加参数dim=d

  > After summing with `dim=d`, the dimension at index `d` of the input is **eliminated** from the shape of the output tensor.
  >
  > 例：x.shape:  torch.Size([3, 4, 5, 6]) 
  >
  > x.sum(dim=0).shape:  torch.Size([4, 5, 6]) 
  >
  > x.sum(dim=1).shape:  torch.Size([3, 5, 6])

  > Some reduction operations return more than one value; for example `min` returns both the minimum value over the specified dimension, as well as the index where the minimum value occurs （具体自己学）

Reduction operations <u>reduce</u> the rank of tensors: the dimension over which you perform the reduction will be removed from the shape of the output. If you pass `keepdim=True` to a reduction operation, the specified dimension will not be removed; the output tensor will instead have a shape of 1 in that dimension.

When you are working with multidimensional tensors, thinking about rows and columns can become confusing; instead it's more useful to think about the shape that will result from each operation.

- matrix operation
  - `torch.dot`: Computes inner product of vectors
  - `torch.mm`: Computes matrix-matrix products
  - `torch.mv`: Computes matrix-vector products
  - `torch.addmm` / `torch.addmv`: Computes matrix-matrix and matrix-vector multiplications plus a bias
  - `torch.bmm`/ `torch.baddmm`: Batched versions of `torch.mm` and `torch.addmm`, respectively
  - `torch.matmul`: General matrix product that performs different operations depending on the rank of the inputs. Confusingly, this is similar to `np.dot` in numpy.

- vectorization

  > In many cases, avoiding explicit Python loops in your code and instead using PyTorch operators to handle looping internally will cause your code to run a lot faster. This style of writing code, called **vectorization**, avoids overhead from the Python interpreter, and can also better parallelize the computation (e.g. across CPU cores, on on GPUs). Whenever possible you should strive to write vectorized code.



**Broadcasting**

自己学（doge



**Out-of-place vs in-place operators**

Most PyTorch operators are classified into one of two categories:

- **Out-of-place operators:** return a new tensor. Most PyTorch operators behave this way.
- **In-place operators:** modify and return the input tensor. Instance methods that end with an underscore (such as `add_()` are in-place. Operators in the `torch` namespace can be made in-place using the `out=` keyword argument.

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
z = x.add(y)  # Same as z = x + y or z = torch.add(x, y)
x.add_(y)  # Same as x += y or torch.add(x, y, out=x)
```

In general, **you should avoid in-place operations** since they can cause problems when computing gradients using autograd



**Running on GPU**

用`device=`参数指定张量存储在哪个设备上  可选'cuda', 'cpu'等

可以用`.to()`来改变张量的存储设备，也可以用`.cuda()`和`.cpu()`
