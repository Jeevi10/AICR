import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from matplotlib import pyplot as plt

#from pushover import notify
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display
import dataloader as dl

import networks
from networks import *
import parameter as p
import misc
import trainer 
from loss_fn_prob import *
from resnet_model import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from class_loss import *
import optuna
import joblib
from pathlib import Path

from os import path  
from contextlib import contextmanager
import multiprocessing
from optuna.samplers import TPESampler
N_GPUS = 4


def train_model():
    
    
    cfg = p.mnist_p
    
    train_loader,test_loader,loader_list = misc.get_dataloaders("Lenet",p.mnist)
    
    
    model = sixNet()
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(f'cuda:0')
    model.load_state_dict(torch.load("./saved_model/model_" + str('pretrain_model_mnist')))
    
    criterion_prox_64 = Proximity(num_classes=p.num_classes, feat_dim= 64, use_gpu=True,margin=cfg['margin_64'], sl=cfg["sl_64"], lamda=cfg["lamda_64"])
    criterion_prox_512 = Proximity(num_classes=p.num_classes, feat_dim=512, use_gpu=True, margin=cfg['margin_512'],sl=cfg["sl_512"], lamda=cfg["lamda_512"])
    criterion_prox_1152 = Proximity(num_classes=p.num_classes, feat_dim=1152, use_gpu=True, margin=cfg['margin_1152'],sl=cfg["sl_1152"], lamda=cfg["lamda_1152"])
    
    criterion_conprox_64 = Con_Proximity(num_classes=p.num_classes, feat_dim= 64, use_gpu=True,margin=cfg['margin_64'], sl=cfg["sl_64"], lamda=cfg["lamda_64"])
    criterion_conprox_512 = Con_Proximity(num_classes=p.num_classes, feat_dim=512, use_gpu=True, margin=cfg['margin_512'],sl=cfg["sl_512"], lamda=cfg["lamda_512"])
    criterion_conprox_1152 = Con_Proximity(num_classes=p.num_classes, feat_dim=1152, use_gpu=True, margin=cfg['margin_1152'],sl=cfg["sl_1152"], lamda=cfg["lamda_1152"])
    criterion_class = MarginLoss(num_classes=p.num_classes, margin=0.995)
    criterion_logit = LogitLoss()
    
    
    
    optimizer_post = torch.optim.Adam([{'params':model.parameters()}], lr=cfg['lr_post'], weight_decay=cfg['reg_weight'])
    optimizer_prox_64 = torch.optim.SGD(criterion_prox_64.parameters(), lr=cfg['lr_prox_64'])
    optimizer_prox_512 = torch.optim.SGD(criterion_prox_512.parameters(), lr=cfg['lr_prox_512'])
    optimizer_prox_1152 = torch.optim.SGD(criterion_prox_1152.parameters(), lr=cfg['lr_prox_1152'])
    
    optimizer_conprox_64 = torch.optim.SGD(criterion_conprox_64.parameters(), lr=cfg['lr_conprox_64'])
    optimizer_conprox_512 = torch.optim.SGD(criterion_conprox_512.parameters(), lr=cfg['lr_conprox_512'])
    optimizer_conprox_1152 = torch.optim.SGD(criterion_conprox_1152.parameters(), lr=cfg['lr_conprox_1152'])
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_post, milestones=[200,250,400], gamma=0.1)
    lr_scheduler_64 = torch.optim.lr_scheduler.MultiStepLR(optimizer_prox_64, milestones=[200,250,400], gamma=0.1)
    lr_scheduler_512 = torch.optim.lr_scheduler.MultiStepLR(optimizer_prox_512, milestones=[200,250,400], gamma=0.1)
    lr_scheduler_1152 = torch.optim.lr_scheduler.MultiStepLR(optimizer_prox_1152, milestones=[200,250,400], gamma=0.1)
    lr_conscheduler_64 = torch.optim.lr_scheduler.MultiStepLR(optimizer_conprox_64, milestones=[200,250,400], gamma=0.1)
    lr_conscheduler_512 = torch.optim.lr_scheduler.MultiStepLR(optimizer_conprox_512, milestones=[200,250,400], gamma=0.1)
    lr_conscheduler_1152 = torch.optim.lr_scheduler.MultiStepLR(optimizer_conprox_1152, milestones=[200,250,400], gamma=0.1)
    
    post_scheduler ={'lr_scheduler':lr_scheduler,
                    'lr_scheduler_64':lr_scheduler_64,
                    'lr_scheduler_512':lr_scheduler_512,
                    'lr_scheduler_1152':lr_scheduler_1152,
                    'lr_conscheduler_64':lr_conscheduler_64,
                    'lr_conscheduler_512':lr_conscheduler_512,
                    'lr_conscheduler_1152':lr_conscheduler_1152,}
    
    post_optimizers = {'optimizer_post': optimizer_post, 'optimizer_prox_64':optimizer_prox_64, 'optimizer_prox_512':optimizer_prox_512,
                      'optimizer_prox_1152':optimizer_prox_1152,'optimizer_conprox_64':optimizer_conprox_64, 'optimizer_conprox_512':optimizer_conprox_512,
                      'optimizer_conprox_1152':optimizer_conprox_1152}
    post_criterions = {'criterion_logit':criterion_logit,'criterion_class':criterion_class,'criterion_prox_64':criterion_prox_64, 'criterion_prox_512':criterion_prox_512, 'criterion_prox_1152':criterion_prox_1152, 'criterion_conprox_64':criterion_conprox_64, 'criterion_conprox_512':criterion_conprox_512, 'criterion_conprox_1152':criterion_conprox_1152 }
    
    if path.exists("./saved_model/"+'model'+"_posttrain_" +'model_mnist_prox' ): 
        print("posttrain model exists.")
    else:
        accuracy = trainer.posttrain(cfg['n_epochs_post'], model,train_loader, test_loader, post_criterions, post_optimizers,post_scheduler,cfg)

    

        
        
if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description = 'pass model name')
    #parser.add_argument('model_name', help='enter the model name')
    #args = parser.parse_args()
    torch.cuda.set_device(0)
    # get all dataloaders
    train_loader, test_loader, train_loader_list = misc.get_dataloaders('Lenet',p.mnist)

    # this will pretrain if pretrained model does not exist, else will skip
    model = sixNet()
    model = nn.DataParallel(model)
    model = model.cuda()
    #model = resnet(num_classes=p.num_classes,depth=110).cuda()
    
    criterion =  nn.CrossEntropyLoss() 
    if path.exists("./saved_model/"+'model'+"_pretrain_" +'model_mnist' ): 
        print("pretrain model exists.")
    else:
        optimizer_pre = torch.optim.Adam([{'params':model.parameters()}], lr=1e-3, weight_decay=1e-5)
        trainer.pretrain(p.pretrain_epochs, model,train_loader, test_loader, criterion, optimizer_pre)
    
    del train_loader
    del test_loader
    
    train_model()

