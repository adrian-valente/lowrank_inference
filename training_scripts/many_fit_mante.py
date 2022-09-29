"""
April 2022.

Reproducing the experiment of fit_mante many times.
"""

import sys
sys.path.append('../')
from low_rank_rnns.modules import *
from low_rank_rnns import mante, helpers, stats

N_REPS = 10  # Number of full-rank networks on which the experiment is performed
N_REREPS = 3  # Number of low-rank nets of each rank that will be fitted to the full-rank one.
N_RANKS = 5

batch = 3
size = 1024
noise_std = 5e-2
alpha = .2
n_epochs = 500
rho = 1.
lr = 1e-3


losses_orig = np.zeros(N_REPS)
accs_orig = np.zeros(N_REPS)
r2s_trunc = np.zeros((N_REPS, N_RANKS))
losses_trunc = np.zeros((N_REPS, N_RANKS))
accs_trunc = np.zeros((N_REPS, N_RANKS))
r2s_fit = np.zeros((N_REPS, N_REREPS, N_RANKS))
losses_fit = np.zeros((N_REPS, N_REREPS, N_RANKS))
accs_fit = np.zeros((N_REPS, N_REREPS, N_RANKS))


for i in range(N_REPS):
    # Train the full-rank net
    x_train, y_train, mask_train, x_val, y_val, mask_val = mante.generate_mante_data(1000)
    net = FullRankRNN(4, size, 1, noise_std, alpha, rho=rho, train_wi=True, train_wo=True)
    train(net, x_train, y_train, mask_train, 30, lr=1e-4, keep_best=True, cuda=1)
    net.to('cpu')
    loss, acc = mante.test_mante(net, x_val, y_val, mask_val)
    losses_orig[i] = loss
    accs_orig[i] = acc
    print(f'loss={loss:.3f}, acc={acc:.3f}')
    torch.save(net.state_dict(), f'../models/mante_many/mante_fr_mantemod_many_{i}_{batch}.pt')

    # Prepare for fitting
    output, traj = net.forward(x_train, return_dynamics=True)
    T = x_train.shape[1]
    target = torch.tanh(traj[:, 1:].detach())
    mask = torch.ones((x_train.shape[0], T, 1))

    _, traj1 = net.forward(x_val, return_dynamics=True)
    traj1 = net.non_linearity(traj1)
    y1 = traj1.detach().numpy().ravel()

    # Truncated networks
    J = net.wrec.detach().numpy()
    u, s, v = np.linalg.svd(J)
    for rank in range(1, N_RANKS+1):
        J_rec = u[:, :rank] * s[:rank] @ v[:rank]
        net3 = FullRankRNN(4, size, 1, 0, alpha, wi_init=net.wi.detach().clone(), wrec_init=torch.from_numpy(J_rec),
                           wo_init=net.wo.detach().clone())
        r2 = stats.r2_nets_pair(net, net3, x_val)
        loss, acc = mante.test_mante(net3, x_val, y_val, mask_val)
        r2s_trunc[i, rank-1] = r2
        losses_trunc[i, rank-1] = loss
        accs_trunc[i, rank-1] = acc

    # Refitted networks
    for j in range(N_REREPS):

        for rank in range(1, N_RANKS+1):
            net2 = LowRankRNN(4, size, size, noise_std, alpha, rank=rank,
                              wo_init=size * torch.from_numpy(np.eye(size)), train_wi=True, train_so=False)
            train(net2, x_train, target, mask, n_epochs, lr=1e-2, clip_gradient=1, keep_best=True, cuda=2)
            net2.to('cpu')
            torch.save(net2.state_dict(), f'../models/mante_many/mante_fitted_r{rank}_{i}_{j}_{batch}.pt')
            out2, traj2 = net2.forward(x_val, return_dynamics=True)

            traj2 = net2.non_linearity(traj2)
            y2 = traj2.detach().numpy().ravel()
            print(y1.shape)
            print(y2.shape)
            r2 = stats.r2_score(y1, y2)
            print(r2)
            r2s_fit[i, j, rank-1] = r2
            # Replace output identity matrix by output vector and compute task performance
            net2.wo = nn.Parameter(net.wo_full.clone())
            net2.output_size = 1
            net2.so = nn.Parameter(torch.tensor([1. * size]))
            loss, acc = mante.test_mante(net2, x_val, y_val, mask_val)
            losses_fit[i, j, rank-1] = loss
            accs_fit[i, j, rank-1] = acc

np.savez(f'../data/mante_many_result_{batch}.npz', losses_orig, accs_orig, r2s_trunc, losses_trunc, accs_trunc, r2s_fit,
         losses_fit, accs_fit)


