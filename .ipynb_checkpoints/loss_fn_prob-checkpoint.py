import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import parameter as p


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
        #d_y_1 = distmat.clone()[mask.clone()]
        #inv_mask = ~mask.clone()
        d_y = distmat.clone()[mask]
        #print("before",d_y)
        d_y = d_y.clamp(min=1e-12, max=1e+12) 
        #print("after",d_y)
        loss = d_y.mean()
        
        return loss , torch.argmin(distmat,dim=1)
    
    
class Con_Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True, margin = 0.0, sl= 0.01 , lamda=0.01):
        super(Con_Proximity, self).__init__()
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
        #d_y_1 = distmat.clone()[mask.clone()]
        inv_mask = ~mask.clone()
        d_c = distmat.clone()[inv_mask]
        d_c = d_c.clamp(min=1e-12, max=1e+12) 
        loss = d_c.mean()
        
        return loss , torch.argmin(distmat,dim=1)
        
        
        
        
        
        
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        
class BarlowTwins(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        #self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        #self.backbone.fc = nn.Identity()
        self.feat_dim = feat_dim
        self.scale_loss = 1 / 32
        self.lambd= 5e-3
        
        # projector
        sizes = [self.feat_dim] + [1024]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(1024, affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        
        Z = data= torch.cat((z1,z2),0)
        #mse_loss = F.mse_loss(z1,z2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(p.batch_size_train)
        #torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return loss, Z #+ mse_loss*0.001, 


        