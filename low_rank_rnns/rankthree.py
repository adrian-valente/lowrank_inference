import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectories(net, inputs, vec1=None, vec2=None, vec3=None, ax=None, labels=None, figsize=None):
    # Getting m1 and m2, orthogonalize basis
    if vec1 is None:
        vec1 = net.m[:, 0].squeeze().detach().numpy()
    if vec2 is None:
        vec2 = net.m[:, 1].squeeze().detach().numpy()
    if vec3 is None:
        vec3 = net.m[:, 2].squeeze().detach().numpy()

    out, traj = net.forward(inputs, return_dynamics=True)
    traj = traj.detach().numpy()

    traj1 = traj @ vec1 / net.hidden_size
    traj1 = traj1.squeeze()
    traj2 = traj @ vec2 / net.hidden_size
    traj2 = traj2.squeeze()
    traj3 = traj @ vec3 / net.hidden_size
    traj3 = traj3.squeeze()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    # xmin = np.min(traj1)
    # xmax = np.max(traj1)
    # ymin = np.min(traj2)
    # ymax = np.max(traj2)
    # adjust_plot(ax, xmin, xmax, ymin, ymax)

    n_trials = inputs.shape[0]
    for i in range(n_trials):
        if labels is not None:
            ax.plot(traj1[i], traj2[i], traj3[i], label=labels[i])
        else:
            ax.plot(traj1[i], traj2[i], traj3[i])

    if labels:
        ax.legend()

    return ax
