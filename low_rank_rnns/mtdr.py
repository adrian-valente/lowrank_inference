import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def mtdr(rates, variables, thresh=.9, plot_svds=False, var_labels=None):
    """
    Apply the mTDR technique (Aoi et al., Nature Neuroscience, 2020) to a neural response dataset.
    :param rates:
    :param variables:
    :param thresh:
    :param plot_svds:
    :param var_labels:
    :return: ranks, Ws, Ss: ranks is a list of the selected rank for each variable, Ws is a list of arrays of shape
    (N_neurons x variable_rank) for each variable containing the identified principal axes of variation on the neuron
    side, Ss is a list of arrays of shape (variable_rank x N_timesteps containing the corresponding temporal patterns.
    """
    K, T, N = rates.shape
    P = variables.shape[1]
    assert variables.shape[0] == K
    regressands = rates.reshape((K, -1))
    lr = LinearRegression().fit(variables, regressands)
    r2 = lr.score(variables, regressands)
    print(f'initial r2={r2}')

    # Factorize the regression matrix for each variable
    ranks = []
    Ws, Ss = [], []
    for i in range(P):
        M = lr.coef_[:, i].reshape((T, N)).T
        u, s, v = np.linalg.svd(M)
        perc_var = s / np.sum(s)
        cutoff = (np.cumsum(perc_var) > thresh).argmax() + 1
        ranks.append(cutoff)
        Ws.append(u[:, :cutoff])
        Ss.append((v[:cutoff].T * s[:cutoff]).T)

        if plot_svds:
            plt.plot([0] + np.cumsum(perc_var).tolist(), marker='o', label=var_labels[i] if var_labels is not None else '')

    # Do it also for the intercept matrix
    M = lr.intercept_.reshape((T, N)).T
    u, s, v = np.linalg.svd(M)
    perc_var = s / np.sum(s)
    cutoff = (np.cumsum(perc_var) > thresh).argmax() + 1
    ranks.append(cutoff)
    Ws.append(u[:, :cutoff])
    Ss.append(v[:cutoff])
    if plot_svds:
        plt.plot([0] + np.cumsum(perc_var).tolist(), marker='o', label='time')
        plt.legend()
        plt.show()
    print('selected ranks: ', ranks)

    # Compute R2 after rank reduction
    lr2 = LinearRegression()
    lr2.coef_ = np.zeros((N * T, P))
    lr2.intercept_ = np.zeros(N * T)
    for i in range(P):
        lr2.coef_[:, i] = (Ws[i] @ Ss[i]).T.reshape((N * T))
    lr2.intercept_ = (Ws[-1] @ Ss[-1]).T.reshape((N * T))
    r2 = lr2.score(variables, regressands)
    print(f'final r2={r2}')

    return ranks, Ws, Ss