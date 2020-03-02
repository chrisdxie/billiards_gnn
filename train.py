
import sys, os
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# tensorboard stuff
from torch.utils.tensorboard import SummaryWriter

# my libraries
import graph_construction as gc
import losses

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Trainer:
    def __init__(self, model_wrapper, config):
        self.model_wrapper = model_wrapper
        self.config = config
        self.setup()

    def setup(self):

        # Initialize stuff
        self.epoch_num = 1
        self.iter_num = 1
        self.infos = dict()

        # Initialize optimizer
        model_params = self.model_wrapper.model.parameters()
        self.optimizer = torch.optim.Adam(model_params, lr=self.config['lr'])

        if self.config['load']:
            self.load(self.config['opt_filename'], self.config['model_filename'])

        # Losses
        self.losses = {
            'node_loss' : losses.NormalizedMSELoss(),
        }

        # Tensorboard stuff
        self.tb_writer = SummaryWriter(self.config['tb_directory'],
                                       flush_secs=self.config['flush_secs'])

    def train(self, num_epochs, data_loader):

        # Some stuff to keep track of
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        end = time()

        # Training mode
        self.model_wrapper.train_mode()

        for epoch_iter in range(num_epochs):
            for batch in data_loader:

                if self.iter_num >= self.config['max_iters']:
                    print("Reached maximum number of iterations...")
                    return

                # Send everything to GPU
                self.model_wrapper.send_batch_to_device(batch)
                curr_state = batch['current_state'] # Shape: [B x n x d]
                next_state = batch['next_state'] # Shape: [B x rollout_num x n x d]
                B, rn, n, d = next_state.shape

                # measure data loading time
                data_time.update(time() - end)

                # Apply the model/multi-step loss
                loss = torch.tensor(0., dtype=torch.float, device=self.model_wrapper.device)
                for i in range(rn):
                    curr_graph = gc.batch_construct_billiards_graph(curr_state)
                    pred_next_graph = self.model_wrapper.model(curr_graph)
                    loss = loss + self.losses['node_loss'](pred_next_graph.x, 
                                                           next_state[:,i,...].reshape(-1,d))
                    curr_state = pred_next_graph.x.reshape(B,n,d)
                loss = loss / rn


                ### Gradient descent ###
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                total_losses.update(loss.item(), B)

                # Record some information about this iteration
                batch_time.update(time() - end)
                end = time()

                # Record information every x iterations
                if self.iter_num % self.config['iter_collect'] == 0:
                    info = {'iter_num': self.iter_num,
                            'Batch Time': round(batch_time.avg, 3),
                            'Data Time': round(data_time.avg, 3),
                            'loss': round(total_losses.avg, 7),
                            }
                    self.infos[self.iter_num] = info

                    # Tensorboard stuff
                    self.tb_writer.add_scalar('Total Loss', info['loss'], self.iter_num)
                    self.tb_writer.add_scalar('Time/per iter', info['Batch Time'], self.iter_num)
                    self.tb_writer.add_scalar('Time/data fetch', info['Data Time'], self.iter_num)

                    # Reset meters
                    batch_time = AverageMeter()
                    data_time = AverageMeter()
                    total_losses = AverageMeter()
                    end = time()

                self.iter_num += 1

            self.epoch_num += 1



    def test(self, data_loader):

        total_losses = 0.

        for batch in data_loader:

            # Run model
            pred_next_graph = self.model_wrapper.run_on_batch(batch)

            # Compute loss            
            next_graph = gc.batch_construct_billiards_graph(batch['next_state'])
            loss = self.losses['node_loss'](pred_next_graph.x, next_graph.x)

            total_losses += loss

        total_losses /= len(data_loader)

        print(f"Total loss: {total_losses:.07f}")


    def save(self, name=None, save_dir=None):
        """ Save optimizer state, epoch/iter nums, loss information

            Also save model state
        """

        # Save optimizer stuff
        checkpoint = {
            'iter_num' : self.iter_num,
            'epoch_num' : self.epoch_num,
            'infos' : self.infos,
        }

        checkpoint['optimizer'] = self.optimizer.state_dict()

        if save_dir is None:
            save_dir = self.config['tb_directory']
        if name is None:
            filename = save_dir + self.__class__.__name__ + '_' \
                                + self.model_wrapper.__class__.__name__ \
                                + '_iter' + str(self.iter_num) \
                                + '_checkpoint.pth'
        else:
            filename = save_dir + name + '_checkpoint.pth'
        torch.save(checkpoint, filename)


        # Save model stuff
        filename = save_dir + self.model_wrapper.__class__.__name__ \
                            + '_iter' + str(self.iter_num) \
                            + '_checkpoint.pth'
        self.model_wrapper.save(filename)

    def load(self, opt_filename, model_filename):
        """ Load optimizer state, epoch/iter nums, loss information

            Also load model state
        """

        # Load optimizer stuff
        checkpoint = torch.load(opt_filename)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded optimizer")

        self.iter_num = checkpoint['iter_num']
        self.epoch_num = checkpoint['epoch_num']
        self.infos = checkpoint['infos']


        # Load model stuff
        self.model_wrapper.load(model_filename)