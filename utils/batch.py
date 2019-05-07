#coding=utf8
import torch

def collate_fn_labeled(batch):
    # batch is a list of Example
    lens = [len(each.word_ids) for each in batch]
    max_len = max(lens)
    word_ids = [each.word_ids + [0] * (max_len - len(each.word_ids)) for each in batch]
    segment_ids = [each.segment_ids + [0] * (max_len - len(each.segment_ids)) for each in batch] 
    labels = [each.label for each in batch]
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    masks = word_ids != 0
    labels = torch.tensor(labels, dtype=torch.long)
    return word_ids, segment_ids, masks, labels

def collate_fn_unlabeled(batch):
    # batch is a list of Example
    lens = [len(each.word_ids) for each in batch]
    max_len = max(lens)
    word_ids = [each.word_ids + [0] * (max_len - len(each.word_ids)) for each in batch]
    segment_ids = [each.segment_ids + [0] * (max_len - len(each.segment_ids)) for each in batch] 
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    masks = word_ids != 0
    return word_ids, segment_ids, masks

def to_device(elements, device=None):
    cuda_elements = []
    for each in elements:
        cuda_elements.append(each.to(device))
    return cuda_elements

