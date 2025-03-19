"""
Reworked on April 2022.

Good lr for 1024 neurons seems to be 1e-5.
"""
from low_rank_rnns.modules import *
from low_rank_rnns import dms

size = 512
noise_std = 5e-2
alpha = .2
rho = .8
lr = 1e-4

dms.delay_duration_max = 500
dms.decision_duration = 50
dms.setup()
x_train, y_train, mask_train, x_val, y_val, mask_val = dms.generate_dms_data(1000)
net = FullRankRNN(2, size, 1, noise_std, alpha, rho=rho, train_wi=True)

train(net, x_train, y_train, mask_train, 100, lr=lr, clip_gradient=.1,  keep_best=True, cuda='0')
# torch.save(net.state_dict(), f'../models/dms_fr_{size}_{rho}_{lr:.0e}.pt')
loss, acc = dms.test_dms(net, x_val, y_val, mask_val)
print(f'loss={loss:.3f}, acc={acc:.3f}')
