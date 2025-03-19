"""
April 2022

Fit low-rank RNN to a low-rank RNN trained on the DMS task (for validation part).
"""

import torch
import numpy as np
from low_rank_rnns import dms
from low_rank_rnns.modules import LowRankRNN
from low_rank_rnns.helpers import map_device

size = 512
noise_std = 5e-2
alpha = .2

x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
net1 = LowRankRNN(2, size, 1, noise_std, alpha, rank=2)
net1.load_state_dict(torch.load(f'../models/dms_rank2_{size}.pt', map_location='cpu'))

output, traj = net1.forward(x_train, return_dynamics=True)
T = x_train.shape[1]
target = torch.tanh(traj[:, 1:].detach())

rank = 2
net2 = LowRankRNN(2, size, size, 0., alpha, rank=rank,
                  wo_init=size * torch.from_numpy(np.eye(size)), train_wi=True, train_so=False)
train(net2, x_train, target, torch.ones((x_train.shape[0], T, 1)), 500, lr=1e-2, clip_gradient=1, keep_best=True, cuda=True)
net2 = net2.cpu()
net2.svd_reparametrization()
torch.save(net2.state_dict(), f'../models/dms_fitlr_{size}.pt')
print(helpers.r2_nets_pair(net1, net2, x_val, rates=True))