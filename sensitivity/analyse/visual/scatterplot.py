import matplotlib.pyplot as plt


def plot_scatter_graph(x_data, y_data, title, x_label, y_label, save=True, filename=None, show=True):

    plt.scatter(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig(filename + ".png")
    if show:
        plt.show()
    # Refresh figure window
    plt.close()
