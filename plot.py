import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def initialize():
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def add_curve(scores, label=None, x=None):
    N = len(scores)
    if x is None:
        x = [int(i+1) for i in range(N)]
    plt.plot(x, scores, label=label)

def add_hline(value, label=None):
    plt.axhline(value, color='black', linestyle='--', label=label)

def save_plot(filename, title=None, x_label=None, y_label=None, legend=True):
    bottom, top = plt.ylim()
    plt.ylim(ymin=min(0, bottom))
    plt.ylim(ymax=max(0, top))
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if legend:
        plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
