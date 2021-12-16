import torch

def in_top_k(targets, preds, k):
    topk = preds.topk(k,largest=False)[1]
    return (targets.unsqueeze(1) == topk).any(dim=1)


def cross_corr(centers):
    c = centers.view(-1,10*centers.size(1))
    corr =torch.matmul(c.T,c)
    loss = torch.norm(torch.triu(corr, diagonal=1, out=None))
    return 2*loss/corr.size(0)