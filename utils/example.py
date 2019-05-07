#coding=utf8
from utils.constants import *

class Example():

    def __init__(self, first, second, label=None):
        super(Example, self).__init__()
        self.first = first
        self.second = second
        self.label = label
        self.pair = self._input_wrapper()

    def _input_wrapper(self):
        return BERT_CLS + self.first + BERT_SEP + self.second + BERT_SEP