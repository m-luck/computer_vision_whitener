# -*- coding: utf-8 -*-

import numpy as np
import torch

target_epsilon = 10**-5
max_episode = 10**5

def generate_function(start_range, end_range, func):
    x = np.arange(start_range,end_range,0.01)   # start,stop,step
    y = func(x)
    return x,  y

class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

# Generate
x, y = generate_function(-np.pi, np.pi, np.cos)
x, y = map(torch.from_numpy, [x,y])
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, len(x), 10, len(y)

# Construct our model by instantiating the class defined above
model = SimpleNet(D_in, H, D_out)


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(max_episode):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x.float())

    # Compute and print loss
    loss = criterion(y_pred, y.float())
    if t % 100 == 99:
        print(t, loss.item())
        if loss.item() < target_epsilon: break

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
