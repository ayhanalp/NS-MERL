import argparse
import datetime
import dateutil.tz
import os

import torch
from torch.autograd import Variable
from torch import nn

import numpy as np


class Tracker():  # Tracker
    """Tracker class to log progress and save metrics periodically

        Parameters:
            save_folder (str): Folder name for saving progress
            vars_string (list): List of metric names to log
            project_string: (str): String decorator for metric filenames

        Returns:
            None
    """
    def __init__(self, save_folder, vars_string, project_string, save_iteration=1, conv_size=1):
        self.vars_string = vars_string
        self.project_string = project_string
        self.dir_name = save_folder

        # [Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]

        self.all_tracker = [[[], 0.0, []] for _ in vars_string]

        self.counter = 0
        self.conv_size = conv_size
        self.save_iteration = save_iteration

        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

    def update(self, updates, generation):
        """Add a metric observed

                Parameters:
                    updates (list): List of new scores for each tracked metric
                    generation (int): Current gen

                Returns:
                    None
        """

        self.counter += 1

        for update, var in zip(updates, self.all_tracker):
            if update == None:
                continue

            var[0].append(update)

        # Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0:
                continue

            var[1] = sum(var[0]) / float(len(var[0]))

        if self.counter % self.save_iteration == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0:
                    continue

                var[2].append(np.array([generation, var[1]]))
                filename = self.dir_name + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')


def compute_stats(tensor, tracker):
    """Computes stats from intermediate tensors

         Parameters:
               tensor (tensor): tensor
               tracker (object): logger

         Returns:
               None


     """
    tracker['min'] = torch.min(tensor).item()
    tracker['max'] = torch.max(tensor).item()
    tracker['mean'] = torch.mean(tensor).item()
    tracker['std'] = torch.std(tensor).item()


def hard_update(target, source):
    """Hard update (clone) from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model

        Returns:
            None
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    # Signature transfer if applicable
    try:
        target.wwid[0] = source.wwid[0]

    except:
	    None


def to_numpy(var):
    """Tensor --> numpy

    Parameters:
        var (tensor): tensor

    Returns:
        var (ndarray): ndarray
    """
    return var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False):
    """numpy --> Variable

    Parameters:
        ndarray (ndarray): ndarray
        volatile (bool): create a volatile tensor?
        requires_grad (bool): tensor requires gradients?

    Returns:
        var (variable): variable
    """

    if isinstance(ndarray, list):
        ndarray = np.array(ndarray)

    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)


def pprint(l):
    """Pretty print

    Parameters:
        l (list/float/None): object to print

    Returns:
        pretty print str
    """

    if isinstance(l, list):
        if len(l) == 0:
            return None

        else:
            ['%.2f'%item for item in l]

    elif isinstance(l, dict):
        return l

    elif isinstance(l, tuple):
        return ['%.2f'%item for item in l]

    else:
        if l == None:
            return None

        else:
            return '%.2f'%l


def list_stat(l):
    """compute average from a list

    Parameters:
        l (list): list

    Returns:
        mean (float): mean
    """
    if len(l) == 0:
        return None

    else:
        arr = np.array(l)
        return '%.2f'%np.min(arr), '%.2f' % np.max(arr), '%.2f' % np.mean(arr), '%.2f' % np.std(arr)


def init_weights(m):
    """Initialize weights using kaiming uniform initialization in place

    Parameters:
        m (nn.module): Linear module from torch.nn

    Returns:
        None
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def list_mean(l):
    """compute average from a list

    Parameters:
        l (list): list

    Returns:
        mean (float): mean
    """
    if len(l) == 0:
        return None

    else:
        return sum(l)/len(l)


def soft_update(target, source, tau):
    """Soft update from target network to source

        Parameters:
              target (object): A pytorch model
              source (object): A pytorch model
              tau (float): Tau parameter

        Returns:
            None

    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)