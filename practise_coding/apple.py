import torch
apple = torch.tensor([100.0,],requires_grad=True)
apple_num = torch.tensor([2.0],requires_grad=True)
apple_price = apple * apple_num
tax = torch.tensor([1.1],requires_grad=True)
price = tax*apple_price
price.backward()
print('price.grad',price.grad)
print('tax.grad',tax.grad)
print('apple_price.grad:',apple_price.grad)
print('apple_num.grad',apple_num.grad)
print('apple.grad',apple.grad)
