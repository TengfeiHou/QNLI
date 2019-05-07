#coding=utf8

def collate_fn_labeled(batch):
    # batch is a list of Example
    inputs = [each.pair for each in batch]
    labels = [each.label for each in batch]

def collate_fn_unlabeled(batch):
    # batch is a list of Example

