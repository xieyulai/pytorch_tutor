# pytorch_tutor

## 导数的性质
- 线性(linearity)
- 乘积法则(product rule)
- 链式法则(chain rule)

## 梯度的性质
- 梯度是表示方向的数值
- 梯度的方向就是函数上升的**方向**,也就是说自变量沿着梯度方向变化的话,函数值就变大
- 梯度的范数越大,函数上述越快

## pytorch的autograd
- 对于输出是标量的函数(scalar valued function),可以显式的求导
- 对于输出是一个向量(vector valued function)的,不能显式求导
- Y = [y1,y2,...,ym]<sup>T</sup>, X = [x1,x2,...,xm]<sup>T</sup>, Y(向量) = f(X), J = Jacobian Matrix 
$$
\begin{matrix}
∂y1/∂x1 & ... & ∂y1/∂xn\\
... & ... & ...\\
∂ym/∂x1 & ... & ∂ym/∂xn\\
\end{matrix}
$$
- l(标量) = g(y),v=(∂l/∂y1,∂l/∂y2,...,∂l/∂ym)<sup>T</sup>
- Jacobian-Vector product 是l关于X的梯度 

