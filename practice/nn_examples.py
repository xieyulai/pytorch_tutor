# -*- coding: utf-8 -*-
"""
PyTorch: nn
-----------
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights.
"""
import torch
 
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 2, 3, 4, 2
 
# Create random Tensors to hold inputs and outputs
#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)
x = torch.tensor( [[-0.2497,  2.0979,  1.7150],        [ 0.6786,  0.4429,  0.7582]])
y = torch.tensor([[-0.0217,  0.8911],        [-1.0743, -1.1462]])
print('x',x)
print('y',y)
# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
print("list model parameters...")
print(list(model.parameters()))  # print(list(model.parameters()))
 
# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
 
learning_rate = 1e-4
for t in range(1):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)
    print('y_pred',y_pred)
 
    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print('iteration',t,"loss_fn", loss.item())
 
    # Zero the gradients before running the backward pass.
    model.zero_grad()
 
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    re = (y_pred-y)
    res = sum(sum(re*re))
 
    print('loss',loss)
    print('re',re)
    print('res',res)
    print("------------------------------------------------------------------")
 
    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    pv = []    # parameter values
    pgrad = [] # parameter values gradient
    pvt = []   # parameter values transposed
    with torch.no_grad():
        for param in model.parameters():            
            pv.append(param)
            pgrad.append(param.grad)
            param -= learning_rate * param.grad
            print('param.grad',param.grad)
            #print('param',param)
 
    # ====================================================================
    # ---- now calculate the back propagation of the network manually ----
    pvt.append(torch.transpose(pv[0],0,1))
    pvt.append(pv[1].view(1,H))
    pvt.append(torch.transpose(pv[2],0,1))
    pvt.append(pv[3].view(1,D_out))
    
    """
    assume the previous network is represented by the following parameters
    x*P0 + P1 = x2
    Relu(x2) = x4
    x4*p2 + p3 = x6
    y_pred == x6
    """
    x1 = torch.matmul(x,pvt[0]) # the same as torch.mm matrix mulplication
    x2 = x1 + pvt[1]
    x3 = (x2>0)
    x3 = x3.float()
    x4 = x3*x2  # dot prduct (inner product)  #don't use torch.mm(x3,x2)
    x5 = torch.matmul(x4,pvt[2]) 
    x6 = x5 + pvt[3] 
    print("-----x6 ", x6)
    print("--y_pred: ", y_pred)
 
    """    
    d(x6)/d(p3) = [1,1,...] ===>>> pgrad[3] = re * [1,1,...]
    d(x6)/d(p2) = x4 ===>>> pgrad[2] = rex4 = re * x4
    """
    re = 2*(x6-y)   # Loss = (y_pred-y)^2, there is no factor 1/2, so the derivative is Loss'=(y_pred-y)
                    # So, there is a 2 factor before the calculation
    res = sum(re)
    print("-----res ", res)
    print("--pgrad[3]: ", pgrad[3])    
    
    rex4 = torch.mm(re.transpose(0,1), x4) # = 2*torch.mm(x4.transpose(0,1), re).transpose(0,1)
    #rer = res.repeat(2).view(2,2).transpose(0,1) # you can test this line if needed
    print("-----rex4 ", rex4)
    print("--pgrad[2]: ", pgrad[2])
 
    """
    in back-proparation: x2 = x4
    d(x2)/d(p1) = [1,1, ....] 
    d(x6)/d(x4) * d(x2)/d(p1) = pvt[2] ===>>> pgrad[1]
    d(x2)/d(p0) = x
    d(x6)/d(x4) * d(x2)/d(p0) = pvt[2]*x ===>>> pgrad[0] = re*pvt[2]*x
    """
    rex2 = torch.mm(re, pvt[2].transpose(0,1))*x3 # dims =(2x5.5x4)*2x4=2x4, * means dot product and . means matrix product
    regx2 = torch.mm(rex2.transpose(0,1), x)      # dims =4x2.2x3=4x3
    print("-----regx2 ", regx2)
    print("--pgrad[0]: ", pgrad[0])
 
    print("-----sum(rex2) ", sum(rex2))
    print("--pgrad[1]: ", pgrad[1])
 
    # set a break point here to see the results
    print("-----------------------")
 
"""
let's view this problem from the dimensios points of view
NOTE: * means dot product and . means matrix product
____________________________________________
func                    |dimension
------------------------|-------------------
x.p0 + p1 = x2          |2x3.3x4 + 2 = 2x4
x4 = relu(x2)           |2x4 ==> 2x4
x4.p2 + p3 = x6         |2x4.4x5 + 2 = 2x5
re = x6- y              |2x5
------------------------|-------------------
___________________________________________________________________________
grad(p2) = re^T.d(x6)/d(p2) = re^T.x4   |5x2.2x4=5x4
grad(p3) = sum_rows(re)                 |2x1
grad(p0) = re.d(x6)/d(x4).d(x2)/d(p0)   |(2x5.4x5^T)^T.2x3 =(4x2).2x3 = 4x3
         = (re.p2^T*Relu)^T.x           | = (4x2).relu.2x3
grad(p1) = re.d(x6)/d(x4).d(x2)/d(p1)   |(2x5.4x5^T)^T.2x3 =(4x2)
         = (re.p2^T*ReLU)^T             |
----------------------------------------|----------------------------------
"""
 
    
 
