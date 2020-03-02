from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
import torch

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class BilliardsDataset(Dataset):

    def __init__(self, config, test=False):
        self.config = config

        if test:
            data = loadmat(config['test_datafile'])
        else:
            data = loadmat(config['train_datafile'])
        self.data = data['y'] # Shape: [N x T x n x 4]

        N, T = self.data.shape[0:2]

        idx_ep, idx_fr = np.meshgrid(list(range(N)),
                                     list(range(T-self.config['rollout_num'])))

        self.idxs = np.reshape(np.stack([idx_ep, idx_fr], 2), (-1, 2), order='F')
        """ Order looks like [[0,0],
                              [0,1],
                              ...
                             ]
        """


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):

        i, j = self.idxs[idx, 0], self.idxs[idx, 1]

        current_state = self.data[i,j] # Shape: [n x 4]
        next_state = self.data[i,j+1:j+1+self.config['rollout_num']] # Shape: [rollout_num x n x 4]

        sample = {'current_state': torch.from_numpy(current_state).float(),
                  'next_state': torch.from_numpy(next_state).float()
                 }

        return sample

    def get_seq(self, seq_num):
        """ Hack to get an entire sequence
        """

        return torch.tensor(self.data[seq_num]).float()

def get_BD_dataloader(config, test=False, batch_size=8, num_workers=4, shuffle=True):

    config = config.copy()
    dataset = BilliardsDataset(config, test=test)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)