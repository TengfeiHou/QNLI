#coding=utf8
import os, sys
from utils.constants import *
from utils.example import Example

def read_dataset(choice='train'):
    assert choice in ['train', 'dev', 'test']
    file_path = DATASETPATH[choice]
    dataset = []
    if choice == 'test':
        with open(file_path, 'r') as infile:
            infile.readline()
            for line in infile:
                line = line.strip()
                idx, first, second = line.split('\t')
                dataset.append(Example(first, second))
    else:
        with open(file_path, 'r') as infile:
            infile.readline()
            for line in infile:
                line = line.strip()
                idx, first, second, label = line.split('\t')
                label = 1 if label == 'entailment' else 0
                dataset.append(Example(first, second, label))
    return dataset

