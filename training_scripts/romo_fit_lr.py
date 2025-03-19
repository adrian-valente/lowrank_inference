"""
April 2022.

Fit low-rank net to another low-rank trained on the Romo task.
"""
from low_rank_rnns.modules import *
from low_rank_rnns import romo, stats

size = 512
noise_std = 5e-2
alpha = .2
n_epochs = 1000
lr = 1e-3

x_train, y_train, mask_train, x_val, y_val, mask_val = romo.generate_data(1000)
net = LowRankRNN(1, size, 1, noise_std, alpha, rank=2)
net.load_state_dict(torch.load('../models/romo_rank2_512.pt'))
loss, acc = romo.test_romo(net, x_val, y_val, mask_val)
print(acc)
print(loss)

output, traj = net.forward(x_train, return_dynamics=True)
n_trials = x_train.shape[0]
T = x_train.shape[1]
target = torch.tanh(traj[:, 1:].detach())

rank = 2
net2 = LowRankRNN(1, size, size, noise_std, alpha, rank=rank, wo_init=size * torch.from_numpy(np.eye(size)), train_wi=True, train_wo=False, train_so=False)
train(net2, x_train, target, torch.ones((n_trials, T, 1)), n_epochs, lr=lr, clip_gradient=1., keep_best=True, cuda=True)
net2 = net2.cpu()
r2 = stats.r2_nets_pair(net, net2, x_val)
print(r2)
torch.save(net2.state_dict(), '../models/romo_fit_lr.pt')

