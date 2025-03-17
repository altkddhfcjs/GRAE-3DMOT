# ------------------------------------------------------------------------
# Modified from QDTrack (https://github.com/SysCV/qdtrack)
# Copyright (c) 2022 Fischer, Tobias and Pang, Jiangmiao and Huang, Thomas E and Qiu, Linlu and Chen, Haofeng and Darrell, Trevor and Yu, Fisher
# ------------------------------------------------------------------------
#  Modified by Hyunseop Kim
# ------------------------------------------------------------------------


import torch
import torch.nn.functional as F


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    """
    calculate simialrity 
    """
    assert method in ['dot_product', 'cosine', "euclidean"]

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'euclidean':
        key_c = key_embeds.size(0)
        # ref_c = ref_embeds.size(0)
        tt = torch.cat((key_embeds, ref_embeds))
        dot_product = torch.mm(tt, tt.t())
        squared_norm = torch.diag(dot_product)

        distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
        distance_matrix = F.relu(distance_matrix)

        mask = (distance_matrix == 0.0).float()
        distance_matrix = distance_matrix + mask * 1e-8
        distance_matrix = torch.sqrt(distance_matrix)
        distance_matrix = distance_matrix * (1.0 - mask)
        distance_matrix = distance_matrix[:key_c, key_c:].float()
        return distance_matrix