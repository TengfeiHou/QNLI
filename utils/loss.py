#coding=utf8
import torch
import torch.nn as nn

def set_nllloss(reduction='none'):
    if float(torch.__version__[:3]) < 0.4 or torch.__version__ == '0.4.0':
        if reduction == 'none':
            loss_function = nn.NLLLoss(reduce=False)
        elif reduction == 'sum':
            loss_function = nn.NLLLoss(size_average=False)
        else:
            loss_function = nn.NLLLoss(size_average=True)
    elif torch.__version__ == '0.4.1':
        if reduction in ['mean', 'elementwise_mean']:
            reduction = 'elementwise_mean'
        loss_function = nn.NLLLoss(reduction=reduction)
    elif torch.__version__ == '1.0.0' or torch.__version__ == '1.0.1':
        if reduction in ['mean','elementwise_mean']:
            reduction = 'mean'
        loss_function = nn.NLLLoss(reduction=reduction)
    else:
        raise ValueError('Unknown pytorch version!')
    return loss_function

def set_bceloss(reduction='none'):
    if float(torch.__version__[:3]) < 0.4 or torch.__version__ == '0.4.0':
        if reduction == 'none':
            loss_function = nn.BCELoss(reduce=False)
        elif reduction == 'sum':
            loss_function = nn.BCELoss(size_average=False)
        else:
            loss_function = nn.BCELoss(size_average=True)
    elif torch.__version__ == '0.4.1':
        if reduction in ['mean', 'elementwise_mean']:
            reduction = 'elementwise_mean'
        loss_function = nn.BCELoss(reduction=reduction)
    elif torch.__version__ == '1.0.0' or torch.__version__ == '1.0.1':
        if reduction in ['mean','elementwise_mean']:
            reduction = 'mean'
        loss_function = nn.BCELoss(reduction=reduction)
    else:
        raise ValueError('Unknown pytorch version!')
    return loss_function

class LabelSmoothingBCE(nn.Module):
    """
        Implement label smoothing BCE loss.
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingBCE, self).__init__()
        if float(torch.__version__[:3]) < 0.4 or torch.__version__ == '0.4.0':
            if reduction == 'none':
                self.criterion = nn.KLDivLoss(reduce=False)
            elif reduction == 'sum':
                self.criterion = nn.KLDivLoss(size_average=False)
            else:
                self.criterion = nn.KLDivLoss(size_average=True)
        elif torch.__version__ == '0.4.1':
            if reduction in ['mean', 'elementwise_mean']:
                reduction = 'elementwise_mean'
            self.criterion = nn.KLDivLoss(reduction=reduction)
        elif torch.__version__ == '1.0.0' or torch.__version__ == '1.0.1':
            if reduction in ['mean','elementwise_mean']:
                reduction = 'mean'
            self.criterion = nn.KLDivLoss(reduction=reduction)
        else:
            raise ValueError('Unknown pytorch version!')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        
    def forward(self, x, target):
        if x.dim() == 2:
            assert x.size(1) == 2
        else:
            neg_x = 1 - x
            x = torch.cat([neg_x.unsqueeze(dim=1), x.unsqueeze(dim=1)], dim=1)
            x = torch.log(x)
        true_dist = torch.zeros_like(x)
        true_dist.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        true_dist += self.smoothing / 2
        true_dist = true_dist.detach()
        return self.criterion(x, true_dist)