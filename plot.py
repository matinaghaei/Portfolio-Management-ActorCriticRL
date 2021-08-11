import matplotlib.pyplot as plt


def plot(scores1, label1, scores2, label2, filename, x=None):
    N = len(scores1)
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Portfolio (Million Dollars)')
    plt.xlabel('Days')
    plt.plot(x, scores1, label=label1)
    plt.plot(x, scores2, label=label2)
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()
