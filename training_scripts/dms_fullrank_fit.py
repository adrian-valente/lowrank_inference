"""
April 2022.

Fitting a full-rank net trained on the DMS task. Works with a rank 2!!
"""

import torch
import numpy as np
from low_rank_rnns import dms
from low_rank_rnns.modules import FullRankRNN
from low_rank_rnns.helpers import map_device

size = 1024
noise_std = 5e-2
alpha = .2
net_id = 2
rank_max = 9
n_epochs = 500

x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
net = FullRankRNN(2, size, 1, noise_std, alpha)
net.load_state_dict(torch.load(f'../models/dms_fr_{size}_0.1_1e-05.pt', map_location='cpu'))
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print(f'loss={loss:.3f}, acc={acc:.3f}')

# Truncated networks
J = net.wrec.detach().numpy()
u, s, v = np.linalg.svd(J)
r2s_trunc = []
losses_trunc = []
accs_trunc = []
for rank in range(1, rank_max):
    J_rec = u[:, :rank] * s[:rank] @ v[:rank]
    net3 = FullRankRNN(2, size, 1, 0, alpha, wi_init=net.wi.detach().clone(), wrec_init=torch.from_numpy(J_rec),
                       wo_init=net.wo.detach().clone())
    r2 = stats.r2_nets_pair(net, net3, x_val)
    r2s_trunc.append(r2)
    loss, acc = dms.test_dms(net3, x_val, y_val, mask_val)
    losses_trunc.append(loss)
    accs_trunc.append(acc)

r2s_fit = []
losses_fit = []
accs_fit = []
# Preparing trajectories for training
output, traj = net.forward(x_train, return_dynamics=True)
T = x_train.shape[1]
target = torch.tanh(traj[:, 1:].detach())
mask = torch.ones((x_train.shape[0], T, 1))
# For testing
out1, traj1 = net.forward(x_val, return_dynamics=True)
traj1 = net.non_linearity(traj1)
y1 = traj1.detach().numpy().ravel()


for rank in range(1, rank_max):
    # Fitting a rank r network
    net2 = LowRankRNN(2, size, size, noise_std, alpha, rank=rank,
                      wo_init=size * torch.from_numpy(np.eye(size)), train_wi=True, train_so=False)
    print(x_train.shape)
    print(mask.shape)
    print(target.shape)
    train(net2, x_train, target, mask, n_epochs, lr=1e-2, clip_gradient=1, keep_best=True, cuda=True)
    net2.to('cpu')
    torch.save(net2.state_dict(), f'../models/dms_matched_r{rank}.pt')
    out2, traj2 = net2.forward(x_val, return_dynamics=True)
    traj2 = net2.non_linearity(traj2)
    y2 = traj2.detach().numpy().ravel()
    r2 = stats.r2_score(y1, y2)
    print(r2)
    r2s_fit.append(r2)
    # Replace output identity matrix by output vector and compute task performance
    net2.wo = nn.Parameter(net.wo_full.clone())
    net2.output_size = 1
    net2.so = nn.Parameter(torch.tensor([1. * size]))
    loss, acc = dms.test_dms(net2, x_val, y_val, mask_val)
    losses_fit.append(loss)
    accs_fit.append(acc)

np.savez(f'../data/dms_results.npz', r2s_trunc, losses_trunc, accs_trunc, r2s_fit, losses_fit, accs_fit)