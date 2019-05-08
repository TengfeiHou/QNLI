#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from utils.constants import *
from utils.loss import *

class BertBinaryClassification(nn.Module):

    def __init__(self, name, dropout=0.1, bce=True, reduction=None, 
        label_smoothing=0.0, bert_grad=False, device=None):
        super(BertBinaryClassification, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL[name])
        self.dropout = nn.Dropout(p=dropout)
        self.bce = bce
        self.label_smoothing = label_smoothing
        if self.label_smoothing > 0:
            self.classifier = nn.Linear(BERT_HIDDEN_SIZE[name], (1 if self.bce else 2))
            self.loss_function = LabelSmoothingBCE(smoothing=label_smoothing, reduction=reduction)
        elif self.bce:
            self.classifier = nn.Linear(BERT_HIDDEN_SIZE[name], 1)
            self.loss_function = set_bceloss(reduction=reduction)
        else:
            self.classifier = nn.Linear(BERT_HIDDEN_SIZE[name], 2)
            self.loss_function = set_nllloss(reduction=reduction)
        self.set_bert_grad(bert_grad)

    def forward(self, word_ids, segment_ids, masks, label):
        _, pooled_output = self.bert(word_ids, segment_ids, masks, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        scores = self.classifier(pooled_output)
        if self.bce:
            logprob = torch.sigmoid(scores).contiguous().view(-1) # bsize
            if not (self.label_smoothing > 0):
                label = label.float()
        else:
            logprob = F.log_softmax(scores, dim=-1) # bsize, 2
        loss = self.loss_function(logprob, label)
        return loss

    def predict(self, word_ids, segment_ids, masks):
        _, pooled_output = self.bert(word_ids, segment_ids, masks, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        scores = self.classifier(pooled_output)       
        if self.bce:
            prob = torch.sigmoid(scores).contiguous().view(-1) # bsize
            neg_prob = 1 - prob
            prob = torch.cat([neg_prob.unsqueeze(dim=1), prob.unsqueeze(dim=1)], dim=1)
            logprob = torch.log(prob)
        else:
            logprob = F.log_softmax(scores, dim=-1) # bsize, 2
        labels = torch.argmax(logprob, dim=1)
        return labels

    def set_bert_grad(self, requires_grad=True):
        for p in self.bert.parameters():
            p.requires_grad = requires_grad
    
    def bert_parameters(self):
        params = []
        for each in self.bert.parameters():
            params.append(each)
        return params

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))



