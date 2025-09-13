---
layout: post
title: understanding DDPM
date: 2025-09-13 21:24:00
description: a brief but strict understanding to DDPM
tags: formatting math
categories: sample-posts
related_posts: false
---
# DDPM：扩散模型梦开始的地方

##  为什么是 DDPM

答案很简单：因为最近扩散模型很火，因此为了蹭上一波热度开始了解扩散模型的方向。DDPM作为一个比较经典的扩散模型，具有较为复杂的数学推导。在学习过程中容易出现理论与代码实践切割的情况。因此这篇文章希望能够统一地介绍DDPM的数学原理以及具体的代码实现。

## 直观感受

### 源自非平衡态热力学的启发

DDPM的核心思想是模仿物理世界中粒子扩散的过程。例如，一滴墨水在水中逐渐散开，最终均匀分布，这是一个从有序到无序的过程。DDPM通过学习这个过程的逆过程，从完全无序的噪声中逐步恢复出有序、清晰的图案。

#### 前向扩散过程

这个过程是非平衡热力学中的熵增过程，系统从有序变得无序。在DDPM中，这个过程表现为一张图像和一个高斯噪声按照一定比例混合，形成新的图像，经过多次上述操作后，我们可以得到一张约为完全噪音的图像。

#### 反向去噪过程

这个过程是前向扩散过程的逆过程，系统从无序变得有序。在DDPM中，这个过程表现为，给模型一张带有噪音的图像，模型尝试生成其中噪音的部分并将其去除。给模型一张纯噪音图像，经过多次去噪，我们就有可能得到一张清晰的图像。

## 数学原理

针对上述直观的过程，我们下面进行较为严格的数学推导。

### 加噪与去噪

我们首先定义时间步 $$T$$ ，这表示我们总共需要加噪，去噪 $$T$$ 次之后才能得到一张完全的噪声图像或生成图像。特别地，我们使用 $$x_t$$ 表示目前的图像被加噪了多少次。$$x_0$$ 表示这是一张没有经过加噪的正常图像，$$x_T$$ 表示这是一张经过 $$T$$ 次加噪后得到的一张噪声图像。

加噪过程的每一步，都只与当前图像有关，而与再之前的图像无关。这说明，加噪过程是一个马尔科夫链。去噪过程我们也假设可以写成一个马尔科夫链。

对于加噪过程，DDPM定义为：

$$
q(x_t|x_{t-1}):=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t \mathbf{I})
$$


对于去噪过程，DDPM 定义为：

$$
p_\theta(x_{t-1}|x_t):=\mathcal{N}（x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
\\
$$

另外，还有加噪轨迹和去噪轨迹：

$$
q(x_{1:T}|x_0):=\prod_{t=1}^Tq(x_t|x_{t-1})
\\p_\theta(x_{0:T}):=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}|x_t)
$$

> [!note]
>
> 其中，$$\{\beta_t\}$$ 是一族超参数，不需要模型学习。去噪过程中高斯分布的均值和方差与图像和时间步相关，在具体实现的时候。DDPM 将 $$\Sigma_\theta$$ 设为了一组只和时间步相关的常数 $$\sigma_t^2$$



### 损失函数

由于我们从噪声生成图像的过程中，希望能够生成给定的图像 $$x_0$$ ，因此我们要计算 $$p_\theta(x_0)$$ :

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})dx_{1:T}
\\
p_\theta(x_{0:T}):=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t)
\\
p_\theta(x_{t-1}|x_t):=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$




模型学习的分布应当尽量与数据集中的分布一致，而我们可以把数据集中每一个图像看作一类。这样，我们可以把 DDPM 看作一个分类模型（虽然数据是连续的）：给定一个高斯噪声，通过不断去噪得到它正确的分类（图像结果），因此我们可以用交叉熵作为损失函数。

$$
\mathbb{E}[-\log p_\theta(x_0)]
$$

然而，在实际过程中我们很难计算 $$p_\theta(x_0)$$ ， 因此通过变分推断近似解决：
$$

\mathbb{E}[-\log p_\theta(x_0)] \le \mathbb{E}_{x_0,q}[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]

\\=\mathbb{E}_{x_0,q}[-\log p(x_T)-\sum_{t \ge 1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}]=:L
$$

> [!note]
>
> 观察不等式的两边，我们注意到，通过变分推断，我们避免了计算难以解析的高维积分，而是通过改为计算 $$\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}$$ 对分布 $$q(x_{1:T|x_0})$$ 的期望来计算 $$\log p_\theta(x_0)$$ 的下界。
>
> 因此，需要注意损失函数实际上要最小化的是两层期望。
>
> [详细证明见附录](#证明附录)

通过变换，$$L$$ 可以被重写为如下形式：

$$
\mathbb{E}_{x_0,q}[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))
\\-\log p_\theta(x_0|x_1)]
$$

期望中从左到右的三项分别称之为 $$L_T, L_{t-1}, L_0$$。这样写的好处在于，我们可以对模型训练的目标有一个清晰而直观的理解：

$$L_T$$ 试图让加噪生成的噪声图像的分布尽量接近于生成图像时采样噪声的分布。在 DDPM 中，由于 $$q$$ 和 $$p(x_T)$$ 实际上都是已知的，因此 $$L_T$$ 是一个可以忽略的常数项。

$$L_{t-1}$$ 试图让可学习的分布 $$p_\theta$$ 去尽量逼近加噪过程的逆过程。这实际上就是在学习预测噪声。

$$L_0$$ 实际上就是负对数化似然。

> [!note]
>
> KL 散度是信息论中的一个概念，用来衡量两个分布之间的差异。
> $$
> D_{KL}(P||Q)=\int P(X)\log \frac{P(X)}{Q(X)}dx
> $$
> 性质：
>
> - 非负性：取零时，当且仅当 $$P=Q$$
> - 不对称性：$$D_{KL}(P||Q) \ne D_{KL}(Q||P)$$
> - 期望形式：$$D_{KL}(P||Q)=\mathbb{E}[\log P(x)-\log Q(x)]$$

> [!important]
>
> 一个令人疑惑的点是：$$L_0$$ 实际上也是预测噪声的一部分，为什么不能并入 $$L_{t-1}$$ 呢？
>
> 问题在于，$$t=1$$ 时的 KL 散度形式为：
> $$
> KL(q(x_0|x_1,x_0)||p_\theta(x_0|x_1))
> $$
> 但是 $$q(x_0|x_1,x_0)$$ 是一个退化分布，退化分布与一半分布的 KL 散度不能写成常规形式，否则会遇到奇异性问题。

### 加噪技巧

由于 DDPM 的性质，加噪是一步一步地进行的。这是否意味着我们如果想得到一张时间步 $$t$$ 的加噪图像，我们真的需要老老实实地加噪 $$t$$ 次呢？NO！实际上我们有：

$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})\mathbf{I})
\\
\alpha:=1-\beta_t
\\
\bar{\alpha_t}:=\prod_{s=1}^t\alpha_s
$$

> [!note]
>
> 根据加噪的定义 $$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t \mathbf{I})$$ ，我们可以将它重新写为递推形式：
> $$
> x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
> \\
> \epsilon \sim \mathcal{N}(0,\mathbf{I})
> $$
> 我们递归地展开一项这个递推式，可以得到：
> $$
> x_{t+1}=\sqrt{\alpha_{t+1}}(\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t)+\sqrt{1-\alpha_{t+1}}\epsilon_{t+1}
> \\
> =\sqrt{\alpha_{t+1}\alpha_t}x_{t-1}+\sqrt{\alpha_{t+1}(1-\alpha_t)}\epsilon_t+\sqrt{1-\alpha_{t+1}}\epsilon_{t+1}
> $$
> 如果一直展开到 $$x_0$$ ，我们有：
> $$
> x_t=\sqrt{\prod_{s=1}^t\alpha_s}x_0+\sum_{s=1}^t\sqrt{(1-\alpha_s)\prod_{j=s+1}^t\alpha_j}\epsilon_s
> $$
> 方差项通过数学归纳法可以发现等同于 $$1-\bar{\alpha}_t$$，均值项显然。

### 去噪技巧

由于加噪的分布和尝试学习的分布都是高斯分布，根据多元高斯 KL 公式，$$L_{t-1}$$ 可以被重写为[如下形式：](#证明附录)

$$
L_{t-1}=\mathbb{E}_q[\frac{1}{2\sigma_t^2}||\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)||^2]+C
$$

> [!note]
>
> 可以算出：
> $$
> q(x_{t-1}|x_t,x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t\mathbf{I})
> \\
> \tilde \mu_t(x_t,x_0):=\frac{\sqrt{\bar \alpha_{t-1}}\beta_t}{1-\bar \alpha_t}x_0+\frac{\sqrt {\alpha_t}(1-\bar \alpha_{t-1})}{1-\bar \alpha_t}x_t
> \\
> \tilde \beta_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar \alpha_t}\beta_t
> $$



> [!important]
>
> 这里可能存在一个疑惑：对 $$q$$ 求期望，但是期望项里并没有出现 $$q$$ 啊？在这里我们可以把 $$q$$ 理解为一次采样过程。其中 $$x_t$$ 和 $$t$$ 就是在时间步 $$t$$ 时的结果。

对于重写的损失形式，一个直觉是我们直接学习 $$\tilde \mu_t$$ 。然而通过进一步展开上述损失，我们可以得到：

$$
L_{t-1}-C=\mathbb E_{x_0,\epsilon}[\frac{1}{2\sigma^2_t}||\frac{1}{\sqrt \alpha_t}(x_t(x_0,\epsilon)-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon)-\mu_\theta(x_t(x_0,\epsilon),t)||^2]
$$

因此在去噪的时候，DDPM 实际上是先学习去噪分布的均值 $$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon)$$，再在这个均值上添加一个随机噪声 $$x_{t-1}=\mu_\theta(x_t,t)+\sigma_t\mathbf z$$，得到去噪图像的。

一种现在更常见的操作是预测噪声：注意到上面对均值的预测中，由于 $$x_t$$ 可以直接作为输入，实际上只有噪声是需要预测的未知量。因此一个motivating 的选择是：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t,t))
$$

将其重新带回重写后的损失函数后可以得到：

$$
L_{t-1}-C=\mathbb{E}_{x_0,\epsilon}[\frac{\beta_t^2}{2\sigma^2_t(1-\bar \alpha_t)}||\epsilon-\epsilon_\theta(\sqrt{\bar \alpha_t}x_0+\sqrt{1-\bar \alpha_t }\epsilon,t)||^2]
$$

这样做的好处是，网络只需要学习一个更为简洁的函数。

> [!note]
>
> $$\epsilon_\theta$$ 是噪声预测器，用来预测噪声，$$\mu_\theta$$ 用来预测去噪结果的均值。在这次重写中我们把一次采样过程分解为了两步：采样原始图像与采样噪声。
>
> 这里有值得注意的一点：$$\mathbf{z}$$ 代表的并不是所谓的噪音，而是一种随机采样。这很像 VAE 中的重参数化技巧：去噪结果实际上也是一个随机变量，是神经网络无法学习的。但是我们可以通过学习这个分布的均值，并随机给这个均值加上一个高斯分布的采样结果，来实现随机化。

## 工程实现

### 数据预处理

#### 数据放缩

DDPM 在原始论文中指出，他们驶使用的图片数据集由 0-255 的整数构成。他们将数据范围从 0-255 线性地缩小到了 [-1,1]

#### 解码器

在反向过程中，由于去噪过程中会出现对高斯分布采样结果的加减，这导致我们最后得到数据不一定能通过数据放缩的逆过程重新转化为 0-255 的整数。因此 ddpm 选择在 $$x_1$$ 进行去噪的过程中，直接计算每一个像素点是 0-255 的概率。解码公式如下：

$$
p_\theta(x_0|x_1)=\prod_{i=1}^D\int_{\delta_\_(x_0^i)}^{\delta_+(x_0^i)}\mathcal{N}(x;\mu_\theta^i(x_1,1),\sigma^2_1)dx
\\
\delta^+(x) =
\begin{cases}
+\infty, & x = 1 \\
x + \tfrac{1}{255}, & x < 1
\end{cases}
\quad \quad
\delta^-(x) =
\begin{cases}
-\infty, & x = -1 \\
x - \tfrac{1}{255}, & x > -1
\end{cases}
$$

公式看上去吓人，但是实际上是一个很直观的结果：在预测完 $$x_0$$ 的均值后，我们得到一个关于这个均值的高斯分布。通过将这个分布在 x 轴等分 256 份，并计算每部分的面积，我们实际上就得到了高斯分布采样落在这每一部分的概率。第 k 份的区间是 $$[k-\frac{1}{255},k+\frac{1}{255}]$$，代表着落在整数 k 附近的概率。这样我们就得到了每个像素点取值的概率分布，我们可以根据不同的选取策略进行实际的赋值。

### 训练过程

```python
for each training step:
    x0 ~ dataset                          # 从数据集中采样图像
    t ~ Uniform({1,...,T})                # 随机选择一个时间步 t
    ε ~ N(0, I)                            # 采样标准高斯噪声
    xt = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*ε  # 添加噪声，alpha_bar_t = ∏_{s=1}^t (1-βs)
    
    # 预测噪声并计算损失
    ε_theta = neural_net(xt, t)
    loss = ||ε - ε_theta||^2               # L2损失
    
    θ = θ - lr * ∇_θ loss                  # 更新网络参数
```
### 生成过程
```python
# 初始化
x_T ~ N(0, I)                             # 从纯高斯噪声开始

# 逐步去噪
for t = T,...,2:                          # 注意最后一步单独处理
    ε_theta = neural_net(x_t, t)          # 预测噪声
    μ_t = 1/sqrt(1-β_t) * (x_t - β_t/sqrt(alpha_bar_t) * ε_theta)  # 均值更新公式
    z ~ N(0, I)                            # 随机高斯噪声
    x_{t-1} = μ_t + sqrt(β_t) * z         # 采样下一步

# 最后一步 t = 1，使用离散解码器生成整数像素
μ_1 = neural_net(x_1, 1)
σ_1 = sqrt(β_1)
x_0 = discretized_decoder(x_1, μ_1, σ_1)  # 高斯积分得到 0-255 的像素值

return x_0                                # 最终生成图像
```

## 接下来要干啥

注意到，在对 DDPM 的原理进行介绍的时候，并没有提到 prompt 的问题。这是因为  DDPM 是一个无条件模型，因此不能人为地控制 DDPM 图像的输出。因此产生了 CDDPM。另外，扩散模型的非平衡态热力学的原理看上去也十分有趣。在~~不~~遥远的未来，我将尝试对这两个方面进行探究。

## 证明附录

### 变分下界证明

$$
\log p_\theta(x_0)=\log \int p_\theta(x_{0:T})dx_{1:T}=\log \int q(x_{1:T}|x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}dx_{1:T}
$$

利用 Jensen 不等式：

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)], \text{where} f\text{ is convex}
$$

令 $$X=\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\sim q, f=\log$$，可以得到：

$$
\log p_\theta(x_0) \ge \mathbb{E}[\log p_\theta(x_{0:T})-\log q(x_{1:T}|x_0)]
$$

将 $$p_\theta$$ 和 $$q$$ 展开得到：

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}|x_t)
\\
q(x_{1:T}|x_0)=\prod_{t=1}^Tq(x_t|x_{t-1})
$$

将这两项带入上面不等式的右侧，可以得到：

$$
\log p_\theta(x_{0:T})-\log q(x_{1:T}|x_0) = - \log p(x_T)-\sum_{t=1}^T \log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}
$$

将结果带入回不等式的右侧，得到我们想要的结果。注意最后不等式的右边是两层期望：先对辅助分布 $$q$$ 求期望，再对 $$x_0$$ 求期望。

### 其它的

鸽了鸽了，写不动了(

