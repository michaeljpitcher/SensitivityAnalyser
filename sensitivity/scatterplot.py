import matplotlib.pyplot as plt


def create_scatter_plot(x_data, y_data, title, filename, x_label, y_label, show=False):
    """
    Create a scatter plot
    :param x_data:
    :param y_data:
    :param title:
    :param filename:
    :param x_label:
    :param y_label:
    :param show:
    :return:
    """
    plt.scatter(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename + ".png")
    if show:
        plt.show()
    # Refresh figure window
    plt.close()