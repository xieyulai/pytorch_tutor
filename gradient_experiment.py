#coding=utf-8

from math import pi
import torch
import torch.optim

x = torch.ones(2,2,requires_grad=True)
print('x = ',x)
f = -x**2+x #函数
print('f = ',f)
print('x_grad=', x.grad)
f.backward(retain_graph = True)#计算梯度
print('x_grad=',x.grad)
print('x = ',x)
f.backward(retain_graph = True)#计算梯度
print('x_grad=',x.grad)
print('f_grad=',f.grad)

