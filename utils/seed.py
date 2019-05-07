#coding=utf8
import random, torch
import numpy as np

def set_random_seed(random_seed=999, device='cpu'):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        if device != 'cuda':
            print("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
        else:
            torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # These two sentences are used to ensure reproducibility with cudnnbackend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
