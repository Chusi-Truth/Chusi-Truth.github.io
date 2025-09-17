---
layout: post
title: 求导与分布的高维视角
date: 2025-09-16 14:22:00
description: 矩阵求导与多元分布
tags: Linear-Algebra Probability-Theory
categories: math
related_posts: false
---
## 为什么要写

在高中阶段，我们学习了一元函数的求导法则与基础的单变量概率分布。但是随着我们进入大学，多元函数以及多元分布成为了工科的重要工具。然而据我所知（不一定正确），在这方面大学生受到的训练较少，在解决实际科研问题的时候基础不牢。因此，本文希望较为系统地总结一下高维视角下的求导与分布。部分公式会有推导，但是不会做严格的证明。

## 矩阵求导法

目前我们学习到的有：标量，向量，矩阵三个可以求导/被求导的元素。在实际应用中，我们常常用到：{上面的九种排列}-{向量对矩阵求导，矩阵对矩阵求导}。这是因为删去的两种排列涉及到了三维和四维的张量。本篇文章将聚焦于：标量对向量求导，

### 标量对向量求导

对于一个标量函数 $y=f(\mathbf{x})$  ，$y$ 的值由 $\mathbf{x}$ 的每一个分量控制。因此每个分量对 $y$ 进行求导为 $\frac{\partial y}{\partial x_i}$。回顾多元函数的微分：$dy=\sum \frac{\partial y}{\partial x_i}x_i$，可以重写为 $dy=[\frac{\partial y}{\partial x_1}...\frac{\partial y}{\partial x_n}]^T[dx_1...dx_n]$.(注：由于书写限制原因，规定不加转置符号的向量为列向量，加转置符号的向量为行向量)，因此，我们可以把求导的结果记为 $[\frac{\partial y}{\partial x_1}...\frac{\partial y}{\partial x_n}]$.

### 向量对向量求导

向量对向量求导可以看作标量对向量求导的扩展。注意到标量对向量求导可以看作一个向量，那么向量与向量相乘可以看作是一个矩阵。自然地拓展出两种形式：

$$
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n}\\
\vdots & & \vdots\\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

$$
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1}\\
\vdots & & \vdots\\
\frac{\partial y_1}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n}

\end{bmatrix}
$$

上面的称为成为分子布局，下面的矩阵称为分母布局。为了书写和理解方便，定义$\frac{\partial y}{\partial x^T}$ 为分子布局，$\frac{\partial y^T}{\partial x}$为分母布局。在机器学习领域，一般使用混合式布局。因此在刚开始学习的时候可能会有“不知道什么时候要转置”的困惑。

### 标量对矩阵求导

标量对矩阵求导也可以看成标量对向量求导的拓展，形式上也十分直观：逐元素对 $y$ 求偏导，并排成和 $X$ 形状相同的矩阵。

$$
\frac{\partial y}{\partial X}=
\begin{bmatrix}
\frac{\partial y}{\partial x_{11}} & \cdots & \frac{\partial y}{\partial x_{1n}}\\
\vdots & & \vdots\\
\frac{\partial y}{\partial x_{m1}} & \cdots & \frac{\partial y_m}{\partial x_{mn}}
\end{bmatrix}
$$

### 统一标量对向量、矩阵求导的形式

上文提到一个由向量作为输入的标量函数的微分可以写成 $dy=[\frac{\partial y}{\partial x_1}...\frac{\partial y}{\partial x_n}]^T[dx_1...dx_n]$ ，我们希望标量对矩阵求导也能写成类似的形式。实际上对于标量对矩阵求导我们有：

$$
dy=\text{tr}[(\frac{\partial y}{\partial X})^TdX]

$$
经过展开验证，我们很容易证明 $(\frac{\partial y}{\partial  X})^TdX$ 的对角线元素 $a_{ij}$ 满足：

$$
a_{ij}=\sum_{k=1}^m(\frac{\partial y}{\partial X})^T_{ik}dX_{kj}=\sum_{k=1}^m(\frac{\partial y}{\partial X})_{ki}dX_{kj}

$$
因此，

$$
dy=\text{tr}[(\frac{\partial y}{\partial X})^TdX]=\sum_{i,j}(\frac{\partial y}{\partial X})_{ij}dX_{ij}
$$

这正是我们求微分的形式。

## 矩阵微分与迹的运算法则

### 矩阵微分的运算法则

实际上，我们往往通过对函数作微分，转换成 $dy=\text{tr}[(\frac{\partial y}{\partial X})^TdX]$ 的形式来求导。因此，了解微分的运算法则极其重要。

$$
d(X+Y)=dX+dY\\
d(XY)=d(X)Y+X(dY)\\
d(X^T)=(dX)^T\\
d(\text{tr}(X))=\text{tr}(dX)\\
d(X \odot Y)=dX \odot Y+X \odot dY\\
d\sigma(X)=\sigma'(X)\odot dX\\
d(X^{-1})=-X^{-1}dXX^{-1}\\
$$

### 迹的运算法则

在计算过程中，给矩阵套上一层 tr 不仅仅式公式形式上的要求，也可以帮助我们在具体计算的时候提供很多帮助。

$$
\text{tr}(A+B)=\text{tr}(A)+\text(tr)B\\
\text{tr}(AB)=\text{tr}(BA)\\
\text{tr}(A)=\text{tr}(A^T)\\
\text{tr}(\text{tr}(A))=\text{tr}(A)\\
\text{tr}((A \odot B)^TC)=\text{tr}(A^T(B \odot C))
$$

### 计算示例

在这一节将通过推导两个经典标量函数的导数。

#### 最小二乘的导数

最小二乘目的最小化：

$$
l=||Ax-y||_2^2=(Ax-y)^T(Ax-y)
$$

以下是一个详细的推导：

$$
dl=d[((Ax-y)^T(Ax-y))]\\
=d(Ax-y)^T(Ax-y)+(Ax-y)^Td(Ax-y)\\
=dx^TA^T(Ax-y)+(Ax-y)^TAdx\\
=\text{tr}(...)=\text{tr}[2(Ax-y)^TAdx]
$$

因此，

$$
\frac{\partial l}{\partial x}=2A^T(Ax-y)
$$

#### 二次型的导数

考虑二次型：

$$
y=x^TAx
$$

以 $x$ 为自变量进行求导：

$$
dy=d(x^TAx)=dx^TAx+x^TAdx=\text{tr}(x^TA^Tdx+x^TAdx)
$$

因此，

$$
\frac{\partial y}{\partial x}=(A+A^T)x
$$

## 多元分布

实际上多元分布的解析形式就是将标量变量、均值换成向量，方差换成协方差矩阵。

### 高斯分布

$$
p(x)=\frac{1}{\sqrt{\text{det}(2\pi\Sigma)}}\text{exp}[\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)]
$$

$$
x=
\begin{bmatrix}
x_a\\
x_b
\end{bmatrix}
\quad
\mu=
\begin{bmatrix}
\mu_a\\
\mu_b
\end{bmatrix}
\Sigma=
\begin{bmatrix}
\Sigma_a & \Sigma_c\\
\Sigma_c^T & \Sigma_b
\end{bmatrix}
$$



#### 边缘分布

$$
p(x_a)=\mathcal{N}_{x_a}(\mu_a,\Sigma_a)\\
p(x_b)=\mathcal{N}_{x_b}(\mu_b,\Sigma_b)
$$

#### 条件分布

$$p(\mathbf{x}_a|\mathbf{x}_b) = \mathcal{N}_{\mathbf{x}_a}(\hat{\mu}_a, \hat{\Sigma}_a) \quad \begin{cases} 
\hat{\mu}_a &= \mu_a + \Sigma_c \Sigma_b^{-1}(\mathbf{x}_b - \mu_b) \\ 
\hat{\Sigma}_a &= \Sigma_a - \Sigma_c \Sigma_b^{-1} \Sigma_c^T 
\end{cases}$$

$$p(\mathbf{x}_b|\mathbf{x}_a) = \mathcal{N}_{\mathbf{x}_b}(\hat{\mu}_b, \hat{\Sigma}_b)\quad \begin{cases} 
\hat{\mu}_b &= \mu_b + \Sigma_c^T \Sigma_a^{-1}(\mathbf{x}_a - \mu_a) \\ 
\hat{\Sigma}_b &= \Sigma_b - \Sigma_c^T \Sigma_a^{-1} \Sigma_c 
\end{cases}$$

#### 线性组合的性质

如果

$$
x \sim \mathcal{N}(\mu_x,\Sigma_x) \quad y \sim \mathcal{N}(\mu_x,\Sigma_y)
$$

那么

$$
Ax+By+c \sim \mathcal{N}(A\mu_x+B\mu_y+c,A\Sigma A^T+B\Sigma_yB^T)
$$

### 其它多元分布

由于其它经典的多元分布不需要矩阵的视角也能理解，因此仅仅简单地写出解析形式。

### 多项分布

$$
p(n|a,N)=\frac{N!}{n_1!\cdots n_d!}\prod_{i=1}^da_i^{n_i}\quad \sum_{i=1}^d n_i=N
$$

### 迪利克雷分布

$$
P(x|\alpha)=\frac{\Gamma(\sum_p^P\alpha_p)}{\prod_p^P\Gamma(\alpha_p)}\prod_p^Px_p^{\alpha_p-1}
$$

## 参考资料

[matrix cookbook]([matrixcookbook.pdf](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf))

[矩阵求导没你想的那么难 - 知乎](https://zhuanlan.zhihu.com/p/25295010816)
