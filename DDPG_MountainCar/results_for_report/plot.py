import numpy as np
import matplotlib.pyplot as pp

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

def plot_stuff():
    labels = ['ucb', 'count', 'param_space', 'ou', 'gauss', 'egreedy_decay', 'egreedy', 'no_exp']
    trials = 5
    epochs = 200
    window = 10
    for i, label in enumerate(labels):
        data = np.load(f'rewards_{label}_5.npy')
        ave = np.average(data, axis=0)
        inv_sqrt_n = 1.0/np.sqrt(trials)
        ave_error = 1.96 * inv_sqrt_n * np.std(data, axis=0)
        ave = rolling_average(data=ave, window_size=window)
        ave_error = rolling_average(data=ave_error, window_size=window)
        upper_bound = ave + ave_error
        lower_bound = ave - ave_error
        pp.plot(range(epochs), ave, label=label)
        pp.fill_between(range(epochs), upper_bound, lower_bound, alpha=0.2)
    pp.title('Continous Mountain Car, DDPG, reward per episode', fontsize=24)
    pp.xlabel('Episode #', fontsize=20)
    pp.ylabel('Reward', fontsize=20)
    pp.ylim(-10,100)
    pp.legend(fontsize=18)
    pp.grid()
    fig = pp.gcf()
    fig.set_size_inches(15,10)
    fig.savefig('results.png')
    # pp.show()

if __name__ == "__main__":
    plot_stuff()