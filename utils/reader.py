#coding=utf8
import os, sys
import torch

def read_dataset(choice='train'):
    assert choice in ['train', 'dev', 'test']
    