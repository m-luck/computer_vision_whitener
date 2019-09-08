import matplotlib.pyplot as plt
import numpy as np
import torch

# Verbose terminal output.
vb = True 

# Open plots.
plot_ = True
independent_plot = False

def load_up_2d_dataset(file_path):
    data = torch.load(file_path)
    if vb: print(type(data))
    coords = data.tolist()

    return coords

def visualize_2d_scatter_plot(coords, zoom):
    x, y = zip(*coords) # Unzip tuple list for clarity.
    plt.scatter(x,y)
    if independent_plot:
        plt.axis([-zoom, zoom, -zoom, zoom], aspect='scaled') # Sets same limits in plots for better comparison.
        plt.show()

def translate_to_zero_mean(coords):
    xs, ys = zip(*coords) # Unzip tuple list.
    x_mean = np.mean(xs) # Mean of each dimension. 
    y_mean = np.mean(ys)
    new_xs = [(lambda x: x - x_mean)(x) for x in xs] # Center around zero by subtracting the mean.
    new_ys = [(lambda x: y - y_mean)(y) for y in ys]

    return new_xs, new_ys
    
def decorrelate_covariance(xs, ys):
    coords = list(zip(xs,ys))
    if vb: print(coords)
    matrix = np.array(coords) # Cast coordinates into a NumPy friendly matrix.
    cov = np.dot(matrix.T, matrix) / matrix.shape[0] # Find the covariance matrix.
    eigenv, singular, unitary = np.linalg.svd(cov) # Singular value decomp.
    matrix_rot = np.dot(matrix, eigenv) # Decorrelate the data.
    
    return matrix_rot, singular
    
def normalize_to_gaussian(decorr_matrix, singular):
    whitened = decorr_matrix / np.sqrt(singular + 1e-5)

    return whitened 

def render(zoom):
    if plot_:
        plt.axis([-zoom, zoom, -zoom, zoom], aspect='scaled') # Sets same limits in plots for better comparison.
        plt.show()

def discuss_dependencies():
    pass

def fit_func_w_NN():
    pass

if __name__ == "__main__":
	def main_whiten():
		coords = load_up_2d_dataset("assign0_data.py")
		if plot_: visualize_2d_scatter_plot(coords, 5)
		x, y = translate_to_zero_mean(coords)
		matrix_rot, singular = decorrelate_covariance(x, y)
		whitened = normalize_to_gaussian(matrix_rot, singular)
		if plot_: visualize_2d_scatter_plot(whitened, 5)
		if plot_: render(5)

	main_whiten()