#coding=utf-8

from math import pi
import torch

x = torch.tensor([1.0],requires_grad=True)
print('x = ',x)
f = -x**2+x #函数
print('f = ',f)
print('x_grad=', x.grad)
f.backward(retain_graph = True)#计算梯度
print('x_grad=',x.grad)
print('x = ',x)
f.backward(retain_graph = True)#计算梯度
print('x_grad=',x.grad)
print('f = ',f)
print('f_grad=',f.grad)

x = torch.randn(3,requires_grad=True)

print(x)
print(x.data)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(v)

print(x.grad)
