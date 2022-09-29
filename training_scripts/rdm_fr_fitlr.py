import sys
sys.path.append('../')

from low_rank_rnns.modules import *
from low_rank_rnns import rdm, helpers, stats

size = 512
noise_std = 5e-2
alpha = .2
n_epochs_fr = 2
n_epochs = 2

x_train, y_train, mask_train, x_val, y_val, mask_val = rdm.generate_rdm_data(1000)
net = LowRankRNN(1, size, 1, noise_std, alpha, rank=1)
train(net, x_train, y_train, mask_train, n_epochs_fr, lr=1e-4, keep_best=True)
# torch.save(net.state_dict(), f'../models/rdm_rank1_{size}.pt')
net.load_state_dict(torch.load(f'../models/rdm_rank1_{size}.pt'))
rdm.test_rdm(net, x_val, y_val, mask_val)

output, traj = net.forward(x_train, return_dynamics=True)
n_trials = x_train.shape[0]
T = x_train.shape[1]
target = torch.tanh(traj[:, 1:].detach())

net2 = LowRankRNN(1, size, size, 0, alpha, rank=1, wo_init=size * torch.from_numpy(np.eye(size)), train_wi=True)
train(net2, x_train, target, torch.ones((n_trials, T, 1)), n_epochs, lr=1e-2, clip_gradient=1., keep_best=True, cuda=True)
net2 = net2.cpu()
# torch.save(net.state_dict(), f'../models/rdm_fitlr_{size}.pt')
out1, traj1 = net.forward(x_val, return_dynamics=True)
out2, traj2 = net2.forward(x_val, return_dynamics=True)
traj1 = net.non_linearity(traj1)
traj2 = net2.non_linearity(traj2)
y1 = traj1.detach().numpy().ravel()
y2 = traj2.detach().numpy().ravel()
r2 = stats.r2_score(y1, y2)
print(r2)



