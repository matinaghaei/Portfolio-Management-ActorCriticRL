import matplotlib.pyplot as plt


def add_curve(scores, label, x=None):
    N = len(scores)
    if x is None:
        x = [i+1 for i in range(N)]
    plt.plot(x, scores, label=label)


def save_plot(filename, x_label, y_label):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
