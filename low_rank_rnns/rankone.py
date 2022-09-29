import matplotlib.pyplot as plt


def plot_trial_averaged_trajectory(net, input, m=None, I=None, ax=None, c='blue', alpha=1, rates=False):
    if m is None:
        m = net.m[:, 0].detach().numpy()
    if I is None:
        I = net.wi[0].detach().numpy()
    print(m.shape)
    print(I.shape)
    I_orth = I - (I @ m) * m / (m @ m)

    output, trajectories = net.forward(input, return_dynamics=True)
    if rates:
        trajectories = net.non_linearity(trajectories)
    trajectories = trajectories.detach().numpy()
    averaged = trajectories.mean(axis=0)
    projection1 = averaged @ m / net.hidden_size
    projection2 = averaged @ I_orth / net.hidden_size

    if ax is None:
        fig, ax = plt.subplots()
    pl, = ax.plot(projection1, projection2, c=c, alpha=alpha, lw=1)
    return pl
