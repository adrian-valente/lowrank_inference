"""
Adrian Valente, May 2022.

Train low-rank networks on the Mante data.
Low-rank networks are pre-initialized with connectivity from a previously trained full-rank.
"""

from low_rank_rnns.modules import *
from low_rank_rnns import mante, stats, data_loader_mante as dlm, helpers
import pandas as pd

bin_width = 5
smoothing_width = 50

hidden_neurons = 0
n_epochs = 2
lr = 1e-2

monkey = 'A'  # Choose monkey
load_modnum = 22  # Numbering of the full-rank fitted network
modnum = 24  # Number of the low-rank fitted networks

# Load preprocessed condition-averaged data (3d tensor, see 2112_mante_monkey_fits.ipynb)
conditions = pd.read_csv(f'../data/conditions_monkey{monkey}.csv')
X = np.load(f'../data/X_cent_monkey{monkey}.npy')
nconds, ntime, n_neurons = X.shape
print(f"Training on monkey {monkey}, version {modnum}")
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

# Prepare training set, initial states for trajectories... (only 2 initial conditions)

# These first lines to set up random train/test split
n_train = int(0.9 * nconds)
idx_train = np.random.choice(np.arange(nconds), n_train, replace=False)
idx_test = np.setdiff1d(np.arange(nconds), idx_train)

# These lines to retrieve saved train/test split
# idx_test = np.load(f'../../data/mante_idx_test_monkey{monkey}_{modnum}.npy')
# idx_train = np.setdiff1d(np.arange(nconds), idx_test)
# n_train = idx_train.shape[0]

# These lines for getting rid of the initial conditions
target = np.concatenate([np.zeros((nconds, mante.ctx_only_pre_duration_discrete, n_neurons)), X], axis=1)
target_torch = torch.from_numpy(target).to(dtype=torch.float32)

mask_train = torch.ones((n_train, inputs.shape[1], n_neurons)).to(dtype=torch.float32)
mask_train[:, :mante.ctx_only_pre_duration_discrete] = 0
# Below: initial states per condition, if desired
# istates = torch.from_numpy(np.concatenate([X_cent[:, 0, :], np.zeros((nconds, hidden_neurons))], axis=1)).to(dtype=torch.float32)
istates = None

# Load full-rank net and check metrics
size = n_neurons + hidden_neurons
net = FullRankRNN(4, size, n_neurons, 0, 0.2, output_non_linearity=(lambda x: x))
net.load_state_dict(torch.load(f'../models/mante_monkey{monkey}_subspace_{load_modnum}.pt'))
_, traj = net(inputs, initial_states=istates, return_dynamics=True)
traj = traj.detach().numpy()[:, :, :n_neurons]
traj_rep = traj[:, mante.ctx_only_pre_duration_discrete+1:, :]
print(X.shape)
print(traj_rep.shape)
print(f'R2 global: {stats.r2_score(X.ravel(), traj_rep.ravel())}')
print(f'R2 per neuron mean: {stats.r2_idneurons(X, traj_rep)}')

# low rank
wrec = net.wrec.detach().numpy()
u, s, v = np.linalg.svd(wrec)
ranks = list(range(1, 6))
for rank in ranks:
    m_init = torch.from_numpy(u[:, :rank] * s[:rank]) * np.sqrt(size)
    n_init = torch.from_numpy(v[:rank].T) * np.sqrt(size)
    net2 = LowRankRNN(4, size, n_neurons, 0, 0.2, rank=rank, train_wo=False, train_wi=True,
                      wo_init=torch.from_numpy(np.diag([1] * n_neurons + [0] * hidden_neurons)[:, :n_neurons]) * size,
                      train_so=False, m_init=m_init, n_init=n_init,
                      output_non_linearity=(lambda x: x))
    train(net2, inputs[idx_train], target_torch[idx_train], mask_train, n_epochs, lr=lr, clip_gradient=1, keep_best=True,
          #     initial_states=istates[idx_train],
          cuda='2')

    net2.cpu()
    _, traj = net2(inputs[idx_test],
                   #initial_states=istates[idx_test],
                   return_dynamics=True)
    traj = traj.detach().numpy()[:, :, :n_neurons]
    traj_rep = traj[:, mante.ctx_only_pre_duration_discrete+1:, :]
    print(f'Rank {rank}, test R2 global: {stats.r2_score(X[idx_test].ravel(), traj_rep.ravel())}')
    print(f'Rank {rank}, test R2 per neuron mean: {stats.r2_idneurons(X[idx_test], traj_rep)}')
    torch.save(net2.state_dict(), f'../models/mante_monkey{monkey}_subspace_rank{rank}_{modnum}.pt')
