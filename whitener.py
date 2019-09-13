import matplotlib.pyplot as plt
import numpy as np
import torch

import shared as sh

# Verbose terminal output.
vb = False 

# Open plots.
plot_ = True
independent_plot = False

def load_up_2d_dataset(file_path):
	data = torch.load(file_path)
	if vb: print(type(data))
	coords = data.tolist()

	return coords

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
	matrix_rot = np.dot(matrix, eigenv) # Decorrelate the data. (Can rotate again if wanted.)
	
	return matrix_rot, singular
	
def normalize_to_gaussian(decorr_matrix, singular):
	whitened = decorr_matrix / np.sqrt(singular + 1e-5) # Just in case singular is zero, add a corrector.

	print('Sanity check: \nVariances should be close to 1.0 --')
	print(np.var(whitened, axis=0))
	print(np.var(whitened))

	return whitened 

def render(zoom):
	if plot_:
		plt.axis([-zoom, zoom, -zoom, zoom], aspect='scaled') # Sets same limits in plots for better comparison.
		plt.show()

if __name__ == "__main__":
	def main_whiten():
		zoom = 5 
		coords = load_up_2d_dataset("assign0_data.py")
		if plot_: orig_plot = sh.visualize_2d_scatter_plot(coords, zoom, False, color='green')
		x, y = translate_to_zero_mean(coords)
		matrix_rot, singular = decorrelate_covariance(x, y)
		whitened = normalize_to_gaussian(matrix_rot, singular)
		if plot_: whitened_plot = sh.visualize_2d_scatter_plot(whitened, zoom, False, color='red')
		plt.legend((orig_plot, whitened_plot),
			('Original', '\'Whitened\''),
			scatterpoints=8,
			loc='upper left',
			ncol=3,
			fontsize=10)
		plt.title("Original vs. 'Whitened' Data")
		plt.xlabel("x")
		plt.ylabel("y")
		if plot_: render(zoom)

	main_whiten(

# Dependencies:

# Although we decorrelated the data, a lack of correlation does not necessarily imply independence. 
# When we plot the data, we see a x-y dependency in the rough form of x^2 + y^2 = c, where c is a constant, a circle.
# So the data does have dependencies.  
# Whitening uses the SVD to transform the variance = 1 on the dimensions.