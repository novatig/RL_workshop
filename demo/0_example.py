
import numpy as np

##################################################
### Importing the library
##################################################

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# torch.set_default_tensor_type(torch.DoubleTensor)

##################################################
### Creating tensors from Python arrays or lists 
### with torch.tensor()
##################################################

data = [1., 2., 3.]
# data_t = torch.tensor(data)
# print(data_t)
# print(data_t.data)
# print(data_t.dtype)

# data_t = torch.DoubleTensor(data)
# print(data_t.dtype)


# # When using the GPU, this has to be taken into account in the tensor declaration (sending data to GPU)
# # data_t = torch.DoubleTensor(data)
# data_t = torch.cuda.DoubleTensor(data)


# data = [[1., 2.], [3., 4.], [5., 6.]]
# data_t = torch.tensor(data)
# print(data_t)
# print(data_t.size())

# data = [[[1., 2.], [3., 4.], [5., 6.]]]
# data_t = torch.tensor(data)
# print(data_t)
# print(data_t.size())

# # # Indexing (get a scalar)
# print(data_t[0])
# print(data_t[0, 0])
# print(data_t[0, 0, 0])
# # Get a Python number
# # print(data_t[0, 0].item())
# # ValueError: only one element tensors can be converted to Python scalars
# temp = np.array(data_t[0, 0])
# print(temp)
# # Get a Python number
# print(data_t[0, 0, 0].item())
# # ValueError: only one element tensors can be converted to Python scalars

# print(data_t.dtype)

# ##################################################
# ### OPERATIONS BETWEEN TENSORS
# ##################################################

# x = torch.tensor([1., 2., 3.])
# y = torch.tensor([4., 5., 6.])
# z = x + y
# print(z)



##################################################
### CONCATENATION BETWEEN TENSORS
##################################################

# # By default, it concatenates along the first axis (concatenates rows)
# x_1 = torch.randn(2, 5)
# y_1 = torch.randn(3, 5)
# z_1 = torch.cat([x_1, y_1])
# print(z_1)
# print(z_1.size())

# # Concatenate columns:
# x_2 = torch.randn(2, 3)
# y_2 = torch.randn(2, 5)
# # second arg specifies which axis to concat along
# z_2 = torch.cat([x_2, y_2], 1)
# print(z_2)
# print(z_2.size())

# # If your tensors are not compatible, torch will complain.  Uncomment to see the error
# # torch.cat([x_1, x_2])





# ##################################################
# ### RESHAPING
# ##################################################

x = torch.randn(2, 3, 4)
# print(x)
# print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
# # Same as above.  If one of the dimensions is -1, its size can be inferred
# print(x.view(2, -1))



##################################################
### BUILDING A COMPUTATION GRAPH
##################################################

# print(x.requires_grad)
# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)
print(x.requires_grad)

# # With requires_grad=True, you can still do all the operations you previously could
# y = torch.tensor([4., 5., 6], requires_grad=True)
# z = x + y
# print(z)
# print(z.requires_grad)

# # BUT z knows something extra.
# print(z.grad_fn)

# # Lets sum up all the entries in z
# s = z.sum()
# print(s)
# print(s.grad_fn)


# # calling .backward() on any variable will run backprop, starting from it.
# s.backward()
# print(x.grad)

















