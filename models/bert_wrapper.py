#coding=utf8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils.constants import *

class BertWrapper(nn.Module):
    

