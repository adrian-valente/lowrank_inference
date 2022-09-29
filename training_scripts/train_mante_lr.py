import sys
sys.path.append('../')

from low_rank_rnns.modules import *
from low_rank_rnns import mante

size = 512
noise_std = 5e-2
alpha = .2
lr = 1e-2

x_train, y_train, mask_train, x_val, y_val, mask_val = mante.generate_mante_data(1000)
net = LowRankRNN(4, size, 1, noise_std, alpha, rank=1, train_wi=True)
net.non_linearity = torch.relu
train(net, x_train, y_train, mask_train, 100, lr=lr, clip_gradient=1., keep_best=True, cuda=True)
torch.save(net.state_dict(), f'../models/mante_rank1_{size}.pt')