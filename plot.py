import matplotlib.pyplot as plt


def add_curve(scores, label, x=None):
    N = len(scores)
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Portfolio (Million Dollars)')
    plt.xlabel('Days')
    plt.plot(x, scores, label=label)


def save_plot(filename):
    plt.legend(loc='best')
    plt.savefig(filename)
