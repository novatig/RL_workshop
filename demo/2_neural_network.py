import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

# TORCH: Always the first dimension is the batch_size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # affine operation
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch size
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# x = torch.randn(3, requires_grad=True)
# params = list(net.parameters())
# print(len(params))
# num_params = [np.prod(temp.size()) for temp in params]
# print(num_params)
# total_params = np.sum(num_params)
# print(total_params)


input = torch.randn(1,1,32,32,requires_grad=True)
out = net(input)
print(out)

net.zero_grad()
error = torch.randn_like(out)
print(error.size())
out.backward(error, retain_graph=True)

grad_input = input.grad
# print(grad_input)
print(input.size())
# print(grad_input.size())



target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)


print(loss.grad_fn)


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

print('conv1.bias before optimizer')
print(net.conv1.bias.data)


# learning rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)


# in your training loop
optimizer.zero_grad() # zero the gradients buffer
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the update

print('conv1.bias after optimizer')
print(net.conv1.bias.data)








