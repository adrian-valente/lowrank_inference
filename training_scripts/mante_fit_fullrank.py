"""
Dec. 2021.

Train a full-rank network on the Mante data.
"""

from low_rank_rnns.modules import *
from low_rank_rnns import mante, stats
import pandas as pd

bin_width = 5
smoothing_width = 50

hidden_neurons = 0
n_epochs = 1000
lr = 5e-4

monkey = 'A'
modnum = 24

# Load preprocessed condition-averaged data (3d tensor, see 2112_mante_monkey_fits.ipynb)
conditions = pd.read_csv(f'../data/conditions_monkey{monkey}.csv')
X = np.load(f'../data/X_cent_monkey{monkey}.npy')
# X = np.load(f'../data/X_zsc_cent_monkey{monkey}.npy')
nconds, ntime, n_neurons = X.shape
print(X.shape)

# Prepare pseudo-inputs
## NEW SETUP: the fitted network is let free for the first 350 ms (while receiving contextual signals).
mante.fixation_duration = 0
mante.ctx_only_pre_duration = 350
mante.stimulus_duration = 650   # counting first step
mante.delay_duration = 80
mante.decision_duration = 20

## SETUP WITHOUT PRE-PERIOD
# mante.fixation_duration = 0
# mante.ctx_only_pre_duration = 0
# mante.stimulus_duration = 645   # adjusted to account for initial state of the network (1st step doesn't count)
# mante.delay_duration = 80
# mante.decision_duration = 20

mante.deltaT = bin_width
mante.SCALE = 1
mante.SCALE_CTX = 1
mante.setup()
correct_trials = conditions.correct == 1
inputs, _, _ = mante.generate_mante_data_from_conditions(conditions[correct_trials]['stim_dir'].to_numpy(),
                                                         conditions[correct_trials]['stim_col'].to_numpy(),
                                                         conditions[correct_trials]['context'].to_numpy())

# Prepare training set, initial states for trajectories...
n_train = int(0.9 * nconds)
idx_train = np.random.choice(np.arange(nconds), n_train, replace=False)
idx_test = np.setdiff1d(np.arange(nconds), idx_train)
X_train = X[idx_train]
X_test = X[idx_test]
inputs_train = inputs[idx_train]
inputs_test = inputs[idx_test]
target_np = np.concatenate([np.zeros((nconds, mante.ctx_only_pre_duration_discrete, n_neurons)), X], axis=1)
target_torch = torch.from_numpy(target_np).to(dtype=torch.float32)
mask_train = torch.ones((n_train, mante.ctx_only_pre_duration_discrete + ntime, n_neurons)).to(dtype=torch.float32)
mask_train[:, :mante.ctx_only_pre_duration_discrete, :] = 0
# istates = torch.from_numpy(np.concatenate([X[:, 0, :], np.zeros((nconds, hidden_neurons))], axis=1)).to(dtype=torch.float32)
istates = None

size = n_neurons + hidden_neurons
net = FullRankRNN(4, size, n_neurons, 0, 0.2, rho=.1, train_wo=False, train_so=False, train_wi=True,
                  wo_init=torch.from_numpy(np.diag([1] * n_neurons + [0] * hidden_neurons)[:, :n_neurons]),
                  output_non_linearity=(lambda x: x))
train(net, inputs_train, target_torch[idx_train], mask_train, n_epochs, lr=lr, clip_gradient=1, keep_best=True,
      #initial_states=istates[idx_train],
      cuda='0')
net.cpu()

_, traj = net(inputs, initial_states=istates, return_dynamics=True)
traj = traj.detach().numpy()[:, :, :n_neurons]
traj_rep = traj[:, mante.ctx_only_pre_duration_discrete+1:, :]
print(f'Original train R2 global: {stats.r2_score(X[idx_train].ravel(), traj_rep[idx_train].ravel())}')
print(f'Original test R2 global: {stats.r2_score(X[idx_test].ravel(), traj_rep[idx_test].ravel())}')
print(f'Original test R2 per neuron mean: {stats.r2_idneurons(X[idx_test], traj_rep[idx_test])}')
torch.save(net.state_dict(), f'../../models/mante_monkey{monkey}_subspace_{modnum}.pt')
np.save(f'../../data/mante_idx_test_monkey{monkey}_{modnum}.npy', idx_test)

