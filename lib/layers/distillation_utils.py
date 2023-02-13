import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def dynamic_entropy_ratio(logits):
    MLP_module = nn.Sequential(nn.Linear(logits.size(0), 1),
                                   nn.Sigmoid()).cuda()
    sigmoid = nn.Sigmoid()

    entropy = - torch.sum(logits* torch.log(logits), dim=1)
    threshold = sigmoid(MLP_module(entropy))

    return threshold  
  
def dynamic_data(logits, selection_strategy, selection_ratio):
    bsz = logits.size(0)

    if selection_strategy == 'entropy':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        _, indices = torch.sort(entropy, descending=True)
        
    elif selection_strategy == 'entropy-r':
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # select most certain data
        _, indices = torch.sort(entropy, descending=False)

    indices = indices[: int(bsz * selection_ratio)]
    
    return indices
