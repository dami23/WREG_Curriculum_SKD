import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.vis_enc import PairEncoder
import pdb, math
       
class Data_Augmentation_Uncertainty_New(nn.Module):
    def __init__(self, opt):
        super(Data_Augmentation_Uncertainty_New, self).__init__()
        self.opt = opt
        self.pair_encoder = PairEncoder(opt)

    def forward(self, pool5, sub_wordembs,  sub_classembs, obj_wordembs, rel_wordembs, ann_pool5, ann_fc7, ann_fleats, scores):
        if scores.size(0) > sub_wordembs.size(0):
            sent_num = scores.size(0)
        else:
            sent_num = sub_wordembs.size(0)

        pair_feats, expand_1_fc7, expand_0_fc7 = self.pair_encoder(pool5, ann_pool5, ann_fc7, ann_fleats, sent_num)
        
        probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # select most certain data
        _, indices = torch.sort(entropy, descending=False)
      
        if scores.size(0) > sub_wordembs.size(0):
            if scores.size(0) > sub_wordembs.size(0) * 2:
                sub_wordembs_rep_0 = sub_wordembs.repeat(scores.size(0) // sub_wordembs.size(0), 1)
                sub_classembs_rep_0 = sub_classembs.repeat(scores.size(0) // sub_wordembs.size(0), 1)
                obj_wordembs_rep_0 = obj_wordembs.repeat(scores.size(0) // sub_wordembs.size(0), 1)
                rel_wordembs_rep_0 = rel_wordembs.repeat(scores.size(0) // sub_wordembs.size(0), 1)

                sub_wordembs_rep_1 = sub_wordembs[:scores.size(0) - sub_wordembs_rep_0.size(0)]
                sub_classembs_rep_1 = sub_classembs[:scores.size(0) - sub_wordembs_rep_0.size(0)]
                obj_wordembs_rep_1 = obj_wordembs[:scores.size(0) - sub_wordembs_rep_0.size(0)]
                rel_wordembs_rep_1 = rel_wordembs[:scores.size(0) - sub_wordembs_rep_0.size(0)]

                sub_wordembs_pro =  torch.cat([sub_wordembs_rep_0,  sub_wordembs_rep_1], 0)[indices]
                sub_classembs_pro =  torch.cat([sub_classembs_rep_0, sub_classembs_rep_1], 0)[indices]
                obj_wordembs_pro =  torch.cat([ obj_wordembs_rep_0,  obj_wordembs_rep_1], 0)[indices]
                rel_wordembs_pro =  torch.cat([rel_wordembs_rep_0, rel_wordembs_rep_1], 0)[indices]

            else:
                inx1 = torch.tensor([i for i in indices if i < sub_wordembs.size(0)])
                inx2 = inx1[:scores.size(0) - sub_wordembs.size(0)]

                sub_wordembs_pro =  torch.cat([sub_wordembs[inx1], sub_wordembs[inx2]], 0)
                sub_classembs_pro =  torch.cat([sub_classembs[inx1], sub_classembs[inx2]], 0)
                obj_wordembs_pro =  torch.cat([obj_wordembs[inx1], obj_wordembs[inx2]], 0)
                rel_wordembs_pro =  torch.cat([rel_wordembs[inx1], rel_wordembs[inx2]], 0)
        
        else:
            sub_wordembs_pro = sub_wordembs[indices]
            sub_classembs_pro = sub_classembs[indices]
            obj_wordembs_pro = obj_wordembs[indices]
            rel_wordembs_pro = rel_wordembs[indices]

        return sub_wordembs_pro, sub_classembs_pro, obj_wordembs_pro, rel_wordembs_pro, pair_feats[indices], expand_1_fc7[indices], expand_0_fc7[indices]
