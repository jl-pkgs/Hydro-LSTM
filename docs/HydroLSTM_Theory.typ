// #import "@preview/physica:0.9.7": *
#import "@preview/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")

// #set page(paper: "a4", margin: 2cm)
// #set text(font: "New Computer Modern", size: 11pt, lang: "zh")
// #set heading(numbering: "1.1")
// #set math.equation(numbering: "(1)")

#align(center, text(size: 20pt, weight: "bold")[Hydro-LSTM 模型实现原理])
#align(center, text(size: 12pt, style: "italic")[
  长短期记忆网络在水文预测中的应用
])

= 1 概述

Hydro-LSTM 是一种基于长短期记忆 (LSTM) 网络的深度学习模型，用于水文时间序列预测。该模型整合了递归神经网络的序列学习能力与传统水文模型的物理直觉，特别是通过"线性地下水库"(linear reservoir) 的门控机制来模拟水文动态过程。

== 1.1 模型架构

完整的 Hydro-LSTM 模型包含两个主要组件：

+ *HydroLSTMCell*：改进的 LSTM 单元，实现了水文过程的门控机制
+ *ModelHydroLSTM*：完整的预测模型，包含 LSTM 层和输出回归层

= 2 HydroLSTMCell：改进的 LSTM 单元

== 2.1 参数定义

HydroLSTMCell 包含以下可训练参数：

- 输入权重：$W_"in" in RR^(d_h times 4 times d_x)$ 或 $RR^(4 times d_x)$
  - 其中 $d_h$ 是隐状态维度 (state_size)
  - $d_x$ 是输入特征维度 (input_size)
  - 因子 4 对应四个门：遗忘门、输入门、输出门、候选状态

- 递归权重：$W_"rec" in RR^(4 times d_h)$
  - 用于计算隐状态的自回归连接

- 偏置项：$b in RR^(4 times d_h)$
  - 为每个门提供偏置

== 2.2 单时间步前向传播 (step_cell)

在第 $t$ 时刻，输入序列中的第 $t$ 个时间步 $x_t in RR^(d_x)$，当前隐状态 $h_(t-1) in RR^(d_h)$，当前细胞状态 $c_(t-1) in RR^(d_h)$。

=== 2.2.1 情形 1：标量隐状态 ($d_h = 1$)

当 $d_h = 1$ 时，计算四个门的值：

$
z_t = W_"rec" dot.c h_(t-1) + b + W_"in" dot.c x_t
$ <eq:gates-scalar>

其中 $z_t in RR^4$ 包含了四个门的预激活值：$z_t = [z_f, z_i, z_o, z_g]^T$。

=== 2.2.2 情形 2：向量隐状态 ($d_h > 1$)

当 $d_h > 1$ 时，为了避免张量对张量的高维操作，我们采用向量化方案。

首先将输入权重重塑：
$
W'_"in" in RR^(4d_h times d_x) = "reshape"(W_"in"), quad "其中" space W_"in" in RR^(d_h times 4 times d_x)
$

计算输入部分：
$
a_1 = W'_"in" dot.c x_t in RR^(4d_h)
$

将其重塑回门级别：
$
A_1 = "reshape"(a_1) in RR^(4 times d_h)
$

使用对角化技巧处理递归连接。将隐状态转换为对角矩阵：
$
H_(t-1) = "diag"(h_(t-1)) in RR^(d_h times d_h)
$

计算递归部分：
$
A_2 = W_"rec" dot.c H_(t-1) in RR^(4 times d_h)
$

合并所有部分得到完整的门值：
$
Z_t = A_2 + b + A_1 in RR^(4 times d_h)
$ <eq:gates-vector>

其中 $Z_t = [z_f; z_i; z_o; z_g]^T in RR^(4 times d_h)$，分别表示遗忘门、输入门、输出门和候选状态的预激活值。

== 2.3 门控机制

从预激活值得到四个门的激活值：

*遗忘门（Forget Gate）*：决定有多少过去的细胞状态被保留
$
f_t = sigma(z_f^t) in (0, 1)^(d_h)
$

*输入门（Input Gate）*：决定有多少新信息被加入细胞状态
$
i_t = sigma(z_i^t) in (0, 1)^(d_h)
$

*输出门（Output Gate）*：决定细胞状态的哪些部分暴露为隐状态
$
o_t = sigma(z_o^t) in (0, 1)^(d_h)
$

*候选状态（Cell Candidate）*：新的潜在信息
$
g_t = tanh(z_g^t) in (-1, 1)^(d_h)
$

其中 $sigma$ 为 Sigmoid 激活函数：
$
sigma(z) = 1/(1 + e^(-z))
$

== 2.4 细胞状态与隐状态更新

=== 2.4.1 情形 1：标量细胞状态 ($d_h = 1$)

细胞状态更新（不应用额外激活）：
$
c_t = f_t dot.o c_(t-1) + i_t dot.o g_t in RR
$ <eq:cell-scalar>

其中 $dot.o$ 表示元素乘法。这个公式体现了 LSTM 的核心思想：通过遗忘门选择性地保留历史信息，通过输入门选择性地加入新信息。

隐状态更新（模拟线性地下水库）：
$
h_t = o_t dot.o tanh(c_t) in RR
$ <eq:hidden-scalar>

=== 2.4.2 情形 2：向量细胞状态 ($d_h > 1$)

对于向量情形，首先将细胞状态转换为对角矩阵以处理多个独立的变量：
$
C_(t-1) = "diag"(c_(t-1)) in RR^(d_h times d_h)
$

细胞状态更新：
$
C_t = f_t * C_(t-1) + (i_t dot.o g_t) in RR^(d_h times d_h)
$ <eq:cell-vector>

其中 $f_t * C_(t-1)$ 表示将遗忘门的每个标量与对角矩阵的对应行相乘，$(i_t dot.o g_t)$ 是按元素乘法的结果。

隐状态更新：
$
h_t = o_t dot.o tanh(c_t) in RR^(d_h)
$ <eq:hidden-vector>

这里 $tanh(c_t)$ 对细胞状态中的每个元素应用 tanh 激活。

== 2.5 时间序列处理

HydroLSTMCell 接收整个时间序列 $X in RR^(d_x times T)$，其中 $T$ 是序列长度（批次大小）。

初始化：
$
h_0, c_0 in RR^(d_h times 1) quad "（通常全零）"
$

对序列中的每个时间步 $t = 1, 2, ..., T$：
$
(h_t, c_t) = "step_cell"(X[:, t], h_(t-1), c_(t-1))
$

收集所有隐状态：
$
H = [h_0, h_1, h_2, ..., h_T] in RR^(d_h times (T+1))
$

*返回值*：
- $H$：完整的隐状态序列
- $h_T$：最后的隐状态（用于状态传递）
- $c_T$：最后的细胞状态（用于状态传递）

= 3 ModelHydroLSTM：完整预测模型

== 3.1 模型组件

完整模型由以下部分组成：

+ *HydroLSTMCell*：核心 LSTM 处理单元
+ *Linear Regression Layer*：线性输出层
+ *State Management*：隐状态和细胞状态的跨批管理

== 3.2 前向传播过程

=== 3.2.1 状态初始化

在第一个 epoch 或状态未初始化时：
$
h_0 = 0 in RR^(d_h times 1), quad c_0 = 0 in RR^(d_h times 1)
$

在后续 epoch 中，使用前一个批次的最终状态（实现截断反向传播 Truncated BPTT）：
$
h_0 = h_"prev" in RR^(d_h times 1), quad c_0 = c_"prev" in RR^(d_h times 1)
$

其中 $h_"prev"$ 和 $c_"prev"$ 由 `Array()` 处理以断开梯度链。

=== 3.2.2 LSTM 前向传播

输入数据 $X in RR^(n times (d_x times T))$ 在转置后变为 $X' in RR^(d_x times (n times T))$，其中 $n$ 是批次大小。

通过 HydroLSTMCell 处理：
$
(H, h_T, c_T) = "HydroLSTMCell"(X', h_0, c_0)
$

其中 $H in RR^(d_h times (T+1))$ 是完整的隐状态序列。

=== 3.2.3 输出回归

应用线性回归层到所有隐状态：
$
Q = W_"out" dot.c H + b_"out" in RR^(1 times (T+1))
$

其中：
- $W_"out" in RR^(1 times d_h)$：输出权重
- $b_"out" in RR$：输出偏置

=== 3.2.4 去掉初始预测

由于初始状态 $h_0$ 是手动初始化的，对应的预测 $Q[:, 1]$ 缺乏物理基础，因此丢弃：
$
Q_"pred" = Q[:, 2:(T+1)] in RR^(1 times T)
$

== 3.3 状态传递与截断反向传播

*状态更新*：每个批次后，更新内部状态以供下一个批次使用：
$
h_"prev" ← "Array"(h_T), quad c_"prev" ← "Array"(c_T)
$

函数 `Array()` 将 Zygote 的反向传播节点转换为普通的 Julia 数组，从而**断开梯度流**。这实现了截断反向传播 (Truncated Backpropagation Through Time, TBPTT)，其优势为：

- 减少反向传播路径长度，避免梯度爆炸/消失
- 降低内存消耗
- 加快训练速度

= 4 完整的前向传播流程图

设输入数据为 $X in RR^(n times d_x times T)$，其中：
- $n = 8$：批次大小
- $d_x = 130$：输入特征维度 (time lags × variables)
- $T = 65$：时间序列长度（滞后步长）

设 $d_h = 4$：隐状态维度

```
输入层 X (8 × 130 × 65)
   ↓ [转置、批处理]
LSTM 输入 X' (130 × 520)  ← 520 = 8 × 65
   ↓
HydroLSTMCell
   ├─ 门控计算：4个门 × 4维 × 520步
   ├─ 细胞状态更新：c_t ∈ ℝ^(4×1)
   ├─ 隐状态更新：h_t ∈ ℝ^(4×1)
   └─ 输出：H (4 × 521)、h_T、c_T
   ↓
线性回归层
   ├─ 权重：W_out (1 × 4)
   ├─ 偏置：b_out ∈ ℝ
   └─ 输出：Q (1 × 521)
   ↓
去掉初始预测
   └─ 输出：Q_pred (1 × 520)
   ↓
状态管理
   ├─ h_prev = Array(h_T)
   ├─ c_prev = Array(c_T)
   └─ 断开梯度链 ↷
```

= 5 数学符号与变量表

#table(
  columns: (1fr, 1fr, 3fr),
  inset: 10pt,
  align: (left, center, left),
  [*符号*], [*维度*], [*含义*],
  [$X$], [$d_x times T$], [输入特征序列],
  [$h_t$], [$d_h times 1$], [第 $t$ 时刻的隐状态],
  [$c_t$], [$d_h times 1$], [第 $t$ 时刻的细胞状态],
  [$f_t$], [$d_h times 1$], [遗忘门激活值],
  [$i_t$], [$d_h times 1$], [输入门激活值],
  [$o_t$], [$d_h times 1$], [输出门激活值],
  [$g_t$], [$d_h times 1$], [候选状态],
  [$W_"in"$], [$d_h times 4 times d_x$ 或 $4 times d_x$], [输入权重],
  [$W_"rec"$], [$4 times d_h$], [递归权重],
  [$b$], [$4 times d_h$], [偏置项],
  [$W_"out"$], [$1 times d_h$], [输出权重],
  [$sigma$], [—], [Sigmoid 激活函数],
  [$tanh$], [—], [双曲正切激活函数],
  [$dot.o$], [—], [按元素乘法],
  [$*$], [—], [矩阵乘法],
)

= 6 训练过程中的关键机制

== 6.1 梯度流与截断反向传播

在标准 LSTM 中，梯度沿着整个序列反向传播：
$
pdv(L, W) = sum_(t=1)^T pdv(L, h_t) dot.c pdv(h_t, W)
$

这会导致梯度路径极长。Hydro-LSTM 通过**状态断开** (state detachment) 进行截断：

$
h_0^((n+1)) = "stopGrad"(h_T^((n)))
$

其中 $n$ 是批次索引。这样，每个批次内的梯度仅在该批次的 $T$ 个时间步内反向传播，显著降低了计算复杂度。

== 6.2 损失函数

常用的损失函数包括：

*1. Huber 损失（平滑L1损失）*：
$
L_"Huber"(hat(y), y) = cases(
  0.5(hat(y) - y)^2 & "if" |hat(y) - y| <= delta,
  delta(|hat(y) - y| - 0.5 delta) & "otherwise"
)
$

其中 $delta$ 是阈值参数（通常为 1）。这个损失函数对异常值鲁棒。

*2. Nash-Sutcliffe 效率系数 (NSE) 损失*：
$
"NSE" = 1 - (sum_t (y_t - hat(y)_t)^2)/(sum_t (y_t - bar(y))^2)
$

其中 $bar(y)$ 是观测值的平均值。水文学中常以最大化 NSE 为目标，因此损失函数定义为：
$
L_"NSE" = -"NSE"
$

== 6.3 权重初始化

所有权重均采用 Xavier (Glorot) Uniform 初始化：
$
W ~ "Uniform"(-sqrt(6/(n_"in" + n_"out")), sqrt(6/(n_"in" + n_"out")))
$

其中 $n_"in"$ 和 $n_"out"$ 分别是输入和输出维度。这确保了深层网络中梯度的均匀分布。

= 7 实现细节

== 7.1 为什么使用对角矩阵？

在向量情形下，对角化 ($h_(t-1) → "diag"(h_(t-1))$) 的目的是将向量-矩阵乘法转换为向量内积的集合：

$
W_"rec" dot.c "diag"(h_(t-1)) = [w_1 dot.o h_(t-1); w_2 dot.o h_(t-1); ...; w_4 dot.o h_(t-1)]^T
$

这使得我们可以对每个门独立地与当前隐状态进行交互，避免了高阶张量运算。

== 7.2 为什么断开梯度链？

`Array()` 函数的目的是将包含自动微分信息的数据结构转换为普通数组，从而：
1. 断开梯度反向传播
2. 释放中间计算图的内存
3. 防止梯度在非常长的序列中爆炸或消失

这等价于 PyTorch 中的 `.detach()` 操作。

== 7.3 为什么丢弃初始预测？

初始隐状态 $h_0$ 是外部初始化的，通常为零向量，不对应于真实的历史信息。因此，根据 $h_0$ 计算的预测 $hat(y)_0$ 缺乏物理基础，应该丢弃。

= 8 超参数设置

在标准配置中：

#table(
  columns: (1fr, 1fr, 2fr),
  inset: 10pt,
  align: (left, center, left),
  [*参数*], [*值*], [*说明*],
  [输入维度 ($d_x$)], [130], [2个变量 × 65步滞后],
  [隐状态维度 ($d_h$)], [4], [LSTM 单元数],
  [时间步长 ($T$)], [65], [序列长度],
  [批次大小], [8], [每批训练样本数],
  [学习率], [$10^(-3)$], [Adam 优化器],
  [Epochs], [50], [完整训练周期],
  [损失函数], [NSE or Huber], [优化目标],
)

= 9 模型性能指标

== 9.1 评估指标

*均方根误差 (RMSE)*：
$
"RMSE" = sqrt(1/n sum_(i=1)^n (y_i - hat(y)_i)^2)
$

*平均绝对误差 (MAE)*：
$
"MAE" = 1/n sum_(i=1)^n |y_i - hat(y)_i|
$

*决定系数 ($R^2$)*：
$
R^2 = 1 - (sum_(i=1)^n (y_i - hat(y)_i)^2)/(sum_(i=1)^n (y_i - bar(y))^2)
$

在研究应用中，通常使用 NSE 作为水文预测模型的标准指标，因为它反映了模型相对于简单平均基线的改进程度。

= 10 参考文献

[1] De la Fuente, L. A., Ehsani, M. R., Gupta, H. V., & Condon, L. E. (2023). Towards Interpretable LSTM-based Modelling of Hydrological Systems. *EGUsphere* [preprint].

[2] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
