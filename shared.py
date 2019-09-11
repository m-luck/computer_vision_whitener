from matplotlib import pyplot as plt

def visualize_2d_scatter_plot(coords, zoom, independent_plot, alpha=1.0, label='None', color='blue'):
    x, y = zip(*coords) # Unzip tuple list for clarity.
    plot_ = plt.scatter(x,y,alpha, label=label, color=color)
    if independent_plot:
        plt.axis([-zoom, zoom, -zoom, zoom], aspect='scaled') # Sets same limits in plots for better comparison.
        plt.show()
    return plot_

