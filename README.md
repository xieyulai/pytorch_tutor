# pytorch_tutor

## 导数的性质
- 线性(linearity)
- 乘积法则(product rule)
- 链式法则(chain rule)

## 梯度的性质
- 梯度是表示方向的数值
- 梯度的方向就是函数上升的**方向**,也就是说自变量沿着梯度方向变化的话,函数值就变大
- 梯度的范数越大,函数上述越快

## 误差的反向传播
- 神经网络的梯度下降指的是对参数的梯度下降
- 梯度下降与误差的反向传播,将最后输出的误差逐层传递下去,遵循链式法则,在每层利用梯度下降更新参数
- 链式法则遵循乘法传递,每一层的梯度相乘
- 误差的反向传播是高效计算权重参数的梯方法度
- 链式法则:复合函数的倒数可以用构成复合函数的各个倒数的乘积表示 

## Batch
- 每一批batch的样本都共同(一起)作用在参数上,而不是一个一个的

## pytorch的autograd
- 对于输出是标量的函数(scalar valued function),可以显式的求导
- 对于输出是一个向量(vector valued function)的,不能显式求导
- Y = [y1,y2,...,ym]<sup>T</sup>, X = [x1,x2,...,xm]<sup>T</sup>, Y(向量) = f(X), J = Jacobian Matrix 
- Jacobian Matrixd 定义如下
$$
\begin{matrix}
∂y1/∂x1 & \cdots & ∂y1/∂xn\\
\vdots & \ddots & \vdots \\
∂ym/∂x1 & \cdots & ∂ym/∂xn\\
\end{matrix}
$$
- l(标量) = g(y),v=(∂l/∂y1,∂l/∂y2,...,∂l/∂ym)<sup>T</sup>
- Jacobian-Vector product 是l关于X的梯度

