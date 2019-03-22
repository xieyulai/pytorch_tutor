import torch
x = torch.tensor([[2.0,3.0],[4.0,9.0]])
y = torch.tensor([[1.0,2.0,4.0],[2.0,4.5,7.0]])
model = torch.nn.Sequential(torch.nn.Linear(2,3))
loss_fn = torch.nn.MSELoss(reduction='sum')
y_pred = model(x)
print('y_pred',y_pred)
print('y',y)
loss = loss_fn(y_pred,y)
loss.backward()
for para in list(model.parameters()):
   print('------')
   print('param',para)
   print('param.grad',para.grad)

