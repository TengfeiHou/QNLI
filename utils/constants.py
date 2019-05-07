#coding=utf8

DATASETPATH = {
    'train': 'data/QNLI/train.tsv',
    'dev': 'data/QNLI/dev.tsv',
    'test': 'data/QNLI/test.tsv'
}

BERT_CLS = '[CLS]'
BERT_SEP = '[SEP]'
BERT_UNK = '[UNK]'
BERT_PAD = '[PAD]'
BERT_MASK = '[MASK]'

BERT_NUM_LAYERS = {
    'bert-base-uncased': 12,
    'bert-large-uncased': 24,
    'bert-base-cased': 12,
    'bert-large-cased': 24,
}
BERT_HIDDEN_SIZE = {
    'bert-base-uncased': 768,
    'bert-large-uncased': 1024,
    'bert-base-cased': 768,
    'bert-large-cased': 1024,   
}

BERT_MODEL = {
    'bert-base-uncased': "data/.cache/bert-base-uncased.tar.gz",
    'bert-large-uncased': "data/.cache/bert-large-uncased.tar.gz",
    'bert-base-cased': "data/.cache/bert-base-cased.tar.gz",
    'bert-large-cased': "data/.cache/bert-large-cased.tar.gz",
}
BERT_VOCAB = {
    'bert-base-uncased': "data/.cache/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "data/.cache/bert-large-uncased-vocab.txt",
    'bert-base-cased': "data/.cache/bert-base-cased-vocab.txt",
    'bert-large-cased': "data/.cache/bert-large-cased-vocab.txt",
}

