import os
import numpy as np
import random
import torch

''' Device type'''
dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_folder(exp_dir, log_dir, model_dir, sample_dir):
    log_root = os.path.join(exp_dir, log_dir)
    model_root = os.path.join(exp_dir, model_dir)
    sample_root = os.path.join(exp_dir, sample_dir)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    if not os.path.isdir(log_root):
        os.makedirs(log_root, exist_ok=True)
    if not os.path.isdir(model_root):
        os.makedirs(model_root, exist_ok=True)
    if not os.path.isdir(sample_root):
        os.makedirs(sample_root, exist_ok=True)

    return log_root, model_root, sample_root


def make_latents(w, batch_size, n_layer, style_mixing_prob=0.9):
    w1, w2 = torch.split(w, batch_size, dim=0)

    w1 = w1.unsqueeze(1).repeat(1, 2 * n_layer - 1, 1)
    w2 = w2.unsqueeze(1).repeat(1, 2 * n_layer - 1, 1)

    layer_idx = torch.from_numpy(np.arange(2 * n_layer - 1)[np.newaxis, :, np.newaxis]).to(dev)
    if random.random() < style_mixing_prob:
        mixing_cutoff = random.randint(1, 2 * n_layer - 1)
    else:
        mixing_cutoff = 2 * n_layer - 1

    dlatents_in = torch.where(layer_idx < mixing_cutoff, w1, w2)
    return dlatents_in