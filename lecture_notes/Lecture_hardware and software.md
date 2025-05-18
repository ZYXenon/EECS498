# Hardware and Software

**CPU vs GPU**

**inside a GPU: RTX Titan**: 

72 Streaming Multiprocessors(SMs)  <--  INT32, 64 FP32 cores per SM, 8 Tensor Core per SM

tensor core: special hardware, optimize matrix multiplication: ABC are 4 by 4 matrix -->  <u>compute AB+C in one clock cycle!</u> （或许可以解释为什么模型大小总是设为2的指数）

**Programming GPUs**: CUDA, OpenCL(similar to CUDA, but runs on anything, slower on NVIDIA hardware)

EECS 598.009: Applied GPU Programming

Usually 8 GPUs per server

**Google Tensor Processing Units (TPU)**: v2, v3  cloud

**a zoo of frameworks**: pytorch, tensorflow, ......   

​	help us : (1)rapidly prototype new ideas (2)automatically compute gradients (3)run efficiently on GPU/TPU

**Computational Graphs**

**PyTorch**

​	fundamental concepts: Tensor, Autograd, Module

​	nn defining modules

​	pretrained models

​	dynamic computation graphs: 

​		every time we run the forward pass, we build a graph, and throw away when backprop

​		allow us to use regular python control flow during the forward pass(比如用if选择哪个权重...)

​		relatively easy to debug

​	static computational graphs

​		build graph describing our computation and reuse the same graph on every iteration

​		JIT

​		with static graphs, framework can optimize the graph before it runs

​		hard to debug  --lots of indirection

​	dynamic vs static   --see slides

**TensorFlow**

**Keras**: high level API

**TensorBoard**: also support pytorch