
import numpy as np

##################################################
### Importing the library
##################################################

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################################################
# Without setting the requires_grad flag, pytorch is not keeping track of the way the tensor is created, not keeping gradients and no backprop
##################################################

x = torch.randn(2, 2)
y = torch.randn(2, 2)

print(x.requires_grad)
print(y.requires_grad)
z = x + y

print(z.grad_fn)

# s = z.sum()
# s.backward()
# # ERROR:
# # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

##################################################
# Change the requires_grad flag IN-PLACE
##################################################

x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
print(z.grad_fn)
z = x + y

# The requires_grad flag is PROPAGATED, so if any of the input arguments of a function has requires_grad=True, so will the output tensor
print(z.grad_fn)
print(z.requires_grad)

##################################################
# DETACHING - detaching a tensor from its history ! Output of tensor.detach() is another tensor sharing the same storage but with forgotten history
##################################################
print(z.grad_fn)

# Can we just take its values, and **detach** it from its history?
new_z = z.detach()

# Now the information to backprop is lost !
print(new_z.grad_fn)
s = z.sum()
s.backward()
print(x.grad)

# s = new_z.sum()
# s.backward()
# ERROR:
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


##################################################
# Temporarily detaching a tensor
##################################################

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


print(x.requires_grad)






    