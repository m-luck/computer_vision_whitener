# -*- coding: utf-8 -*-

import numpy as np
import torch

from matplotlib import pyplot as plt

import shared as sh

vb = True # Verbose.

target_epsilon = 10**-5
max_episode = 10**5

def generate_bounded_function(start_range, end_range, func):
    x = np.arange(start_range,end_range,0.01)   # start,stop,step
    y = func(x)
    return x,  y

class SimpleNet(torch.nn.Module):
    """From PyTorch docs"""
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables. We also implement a tanh activation function.
        """
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.activation = torch.nn.Tanh() # The only nonlinearity will be because of tanh activation on the first layer.
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_tanh = self.activation(self.linear1(x))
        y_pred = self.linear2(h_tanh) 
        return y_pred

# Generate
x, y = generate_bounded_function(-np.pi, np.pi, np.cos)
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

y_untrained = model(x.float())

for t in range(max_episode):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x.float()) # Cast x to float tensor for PyTorch.

    # Compute and print loss
    loss = criterion(y_pred, y.float()) # Cast y to floats for PyTorch.
    if t % 100 == 99:
        print(t, loss.item())
        if loss.item() < target_epsilon: break

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y, y_pred = map(lambda var: var.detach().numpy(), [y, y_pred]) # Detach from gradients.
print(y)

print('Sanity check: ')
print(np.corrcoef(y, y_pred)[0,1]) # Check that the network is well trained.
print('Correlation coefficient should be close to 1.')

# Map a function to be able to plot the respective graphs.
ground, pred, untrained = map(lambda tup: list(zip(tup[0].tolist(), tup[1].tolist())), [(x, y), (x, y_pred), (x, y_untrained)])
if vb: print(ground, '\n', pred)

ground_p = sh.visualize_2d_scatter_plot(ground, 4, False, 0.7, 'Ground', 'red')
pred_p = sh.visualize_2d_scatter_plot(pred, 4, False, 0.2, 'Prediction', 'blue')
untrained_p = sh.visualize_2d_scatter_plot(untrained, 4, False, 1.0, 'Untrained', 'green')

plt.legend((ground_p, pred_p, untrained_p),
           ('Ground', 'Prediction', 'Untrained'),
           scatterpoints=8,
           loc='upper left',
           ncol=3,
           fontsize=10)
plt.title("Ground Truth, Trained Network Prediction, & Untrained Predictions (of y=cos(x))")
plt.xlabel("x")
plt.ylabel("y")
plt.show()