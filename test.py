#coding=utf-8
#import torch
#t2 = torch.tensor([[0,1,2],[3,4,5]])
#print(t2)
#print('data= {}'.format(t2))
#print(t2.reshape(3,2))
#print(t2+1)

from math import pi
import torch
x = torch.tensor([pi/3,pi/6],requires_grad=True)#自变量
print('x = {}'.format(x))
f = -((x.cos() **2).sum())**2 #函数
print('value = {} ' .format(f))
f.backward()#梯度,梯度是一个方向,沿着梯度方向改变自变量
print('grad = {}'.format(x.grad))


