#coding=utf8
import torch.optim as optim

def set_adam_optimizer(params, lr, weight_decay=1e-5):
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay) # (beta1, beta2)
    return optimizer