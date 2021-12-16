import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import parameter as p


def center_corr(centers, indices, batch_size, feature_size,lamda):
    c = centers.unsqueeze(dim=0).expand(batch_size,p.n_classes,feature_size)
    indices_1 = indices[:,0].unsqueeze(-1)
    indices_1 = indices_1.repeat(1,feature_size)
    indices_1 = indices_1.unsqueeze(1)
    
    indices_2 = indices[:,1].unsqueeze(-1)
    indices_2 = indices_2.repeat(1,feature_size)
    indices_2 = indices_2.unsqueeze(1)
    
    
    mat = torch.gather(c, 1, indices_1 , sparse_grad=False, out=None)
    mat_1 = torch.gather(c, 1, indices_2 , sparse_grad=False, out=None)
    
    corr = torch.matmul( mat.transpose(2,1),mat_1)
    on_diag = torch.diagonal(corr).pow_(2).sum().mul(1/32)
    off_diag = torch.triu(corr, diagonal=1, out=None).pow(2).sum(-1).sum(-1).sum(-1)*1/32
    
    loss = on_diag + lamda*off_diag
    
    
    return loss



class Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True, margin = 0.0, sl= 0.01 , lamda=0.01):
        super(Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.margin = margin
        self.sl = sl
        self.lamda = lamda
        
        if self.use_gpu:
            self.centers =  nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
    def forward(self, x , labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        d_y = distmat[mask.clone()]
        inv_mask = ~mask.clone()
        d_y = d_y.unsqueeze(-1)
        #print((distmat*inv_mask).shape)
        d_c = distmat[inv_mask]
        values, indices = torch.topk(distmat,2, dim=1, largest=False, sorted=True, out=None)
        #d_y = d_y.repeat(1,self.num_classes-1)
        d_c = d_c.view(batch_size,self.num_classes-1)
        d_c = torch.mean(d_c,dim=1)
        #d_1 = values[:,0]
        #d_2 = values[:,1]
        #d_3 = values[:,2]
        #d_4 = values[:,3]
        #assert(d_y.shape==d_c.shape)
        #indicators = utils.in_top_k(labels,distmat,1)[:,0]
        #con_indicators = ~ indicators.clone()
    
        #d_c = d_2*indicators + d_1*con_indicators
        
        loss = F.relu((d_y-d_c)/(d_y+d_c) + self.margin)
        #loss = loss.mean(dim=1)
        loss_corr = center_corr(self.centers, indices, batch_size, self.feat_dim, self.lamda)
        mean_loss = loss.mean()
        final_loss = mean_loss #+ loss_corr*self.sl
        return final_loss , torch.argmin(distmat,dim=1)
        