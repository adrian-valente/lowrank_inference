import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from low_rank_rnns.helpers import flatten_trajectory
from low_rank_rnns.stats import pca_fit, cvPCA

#%% SMALL UTILITIES

def setup_matplotlib():
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 19
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['axes.titlepad'] = 24
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 'medium'


def adjust_plot(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin - 0.05 * (xmax - xmin),
                xmax + 0.05 * (xmax - xmin))
    ax.set_ylim(ymin - 0.05 * (ymax - ymin),
                ymax + 0.05 * (ymax - ymin))


def set_size(size, ax=None):
    """ to force the size of the plot, not of the overall figure, from
    https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    w, h = size
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def center_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.set(xticks=[], yticks=[])


def remove_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set(xticks=[], yticks=[])


def center_limits(ax):
    xmin, xmax = ax.get_xlim()
    xbound = max(-xmin, xmax)
    ax.set_xlim(-xbound, xbound)
    ymin, ymax = ax.get_ylim()
    ybound = max(-ymin, ymax)
    ax.set_ylim(-ybound, ybound)


#%% SPECIALIZED PLOTS

def eigenvalue_plot(W, ax=None, figsize=(4, 4), c='lightslategray'):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    eigvals = np.linalg.eigvals(W)
    ax.scatter(np.real(eigvals), np.imag(eigvals), c=c)
    ax.set_xlabel(r'$\Re(\lambda)$')
    ax.set_ylabel(r'$\Im(\lambda)$')
    ax.add_artist(Circle((0, 0), 1, fill=False, ec='k'))
    ax.set_aspect(1)


def pca_cumvar(X, n_components=20, label=None, cross_validate=False):
    X = flatten_trajectory(X)
    print(X.shape)
    if cross_validate:
        expvars, optd, optv, V = cvPCA(X, n_components=n_components)
        plt.plot(np.arange(n_components + 1), np.concatenate([[0], expvars]), marker='o', label=label)
        plt.ylim(0, 1.05)
        return {'expvar': expvars, 'optd': optd, 'optv': optv, 'V': V}
    else:
        pca = pca_fit(X, n_components)
        plt.plot(np.arange(n_components + 1), np.concatenate([[0], np.cumsum(pca.explained_variance_ratio_)]), marker='o',
                label=label)
        plt.ylim(0, 1)
        return pca