import torch.nn as nn
import torch
import torch.nn.functional as F

def CELoss(inputs, targets, ignore=None):
    if inputs.shape[1] == 1:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
        ce_loss = F.cross_entropy(inputs, torch.argmax(targets, axis=1), reduction='none')

    if ignore is not None:
        ignore = 1 - ignore.squeeze()
        ce_loss = ce_loss * ignore

    return ce_loss.mean()

def FocalTverskyLoss(inputs, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=1, ignore=None):
    if inputs.shape[1] == 1:
        inputs = torch.sigmoid(inputs)
    else:
        inputs = torch.softmax(inputs, dim=1)

    if ignore is None:
        tp = (inputs * targets).sum(dim=(0, 2, 3))
        fp = (inputs).sum(dim=(0, 2, 3)) - tp
        fn = (targets).sum(dim=(0, 2, 3)) - tp
    else:
        ignore = (1-ignore).expand(-1, targets.shape[1], -1, -1)
        tp = (inputs * targets * ignore).sum(dim=(0, 2, 3))
        fp = (inputs * ignore).sum(dim=(0, 2, 3)) - tp
        fn = (targets * ignore).sum(dim=(0, 2, 3)) - tp
    
    ft_score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft_loss = (1 - ft_score) ** gamma
    
    return ft_loss.mean()

class CustomLoss(nn.Module):
    __name__ = 'hardnetmseg_loss'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, mask):
        polyp_mask, neo_mask = split_mask(mask)
        
        
        ce_loss = CELoss(y_pr, mask, ignore=None)
        ft_loss = FocalTverskyLoss(y_pr, mask, ignore=None)
        main_loss = ce_loss + ft_loss

        return main_loss