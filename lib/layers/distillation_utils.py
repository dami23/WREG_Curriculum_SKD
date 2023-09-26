import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def dynamic_data(logits, selection_strategy, selection_ratio):
    bsz = logits.size(0)

    if selection_strategy == 'entropy':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        _, indices = torch.sort(entropy, descending=True)
        
    elif selection_strategy == 'entropy-r':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # select most certain data .ceil()
        _, indices = torch.sort(entropy, descending=False)
    
    inx_num = int(bsz * selection_ratio)
    indices_h = indices[: inx_num]
    indices_l = indices[inx_num :]

    return indices_h, indices_l

def dynamic_ratio(logits):
    probs = F.softmax(logits, dim=-1)  
    entropy = - torch.sum(probs * torch.log(probs), dim=1)
    avg_prob = 1 /logits.size(1) * torch.ones((1, logits.size(1)))
    threshold = - entropy / torch.sum(avg_prob * torch.log(avg_prob))

    return threshold.mean()
