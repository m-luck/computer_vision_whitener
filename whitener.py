import matplotlib.pyplot as plt
import numpy as np
import torch


# Verbose terminal output.
vb = True 

# Open plots.
plot_ = False

def load_up_2d_dataset():
    file_path = "assign0_data.py"
    data = torch.load(file_path)
    if vb: print(type(data))
    coords = data.tolist()
    return coords

def visualize_2d_scatter_plot(coords):
    x, y = zip(*coords)
    plt.scatter(x,y)
    plt.show()

def translate_to_zero_mean(coords):
    xs, ys = zip(*coords)
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    new_xs = [(lambda x: x - x_mean)(x) for x in xs]
    new_ys = [(lambda x: y - y_mean)(y) for y in ys] # Could make into function but clearer this way.
    return new_xs, new_ys
    
def decorrelate_covariance_to_identity():
    pass

def plot_whitened_data():
    pass

def discuss_dependencies():
    pass

def fit_func_w_NN():
    pass

coords = load_up_2d_dataset()
if plot_: visualize_2d_scatter_plot(coords)
new_coords = translate_to_zero_mean(coords)