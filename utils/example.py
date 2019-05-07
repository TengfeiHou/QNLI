#coding=utf8
from utils.constants import *
from pytorch_pretrained_bert import BertTokenizer

class Example():

    _tokenizer = None

    def __init__(self, first, second, label=None):
        super(Example, self).__init__()
        self.first = first.strip()
        self.second = second.strip()
        self.label = label
        self.pair = self._bert_input_wrapper()
        self.word_pieces = 
        self.word_ids = self._convert_tokens_to_ids()

    @classmethod
    def set_tokenizer(cls, name):
        cls._tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB[name], do_lower_case=('uncased' in name))

    def _bert_input_wrapper(self):
        return BERT_CLS + ' ' + self.first + BERT_SEP + self.second + ' ' + BERT_SEP