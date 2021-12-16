import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from loss_fn_prob import *
import parameter as p
import tester
import copy
import utils 
import numpy as np
from vis_help import extract_embeddings
from pathlib import Path
#from plot import *


def add_noise(inputs):
    noise = torch.randn_like(inputs).cuda().sign()
    eps = np.random.uniform(0.1,0.3)
    output = inputs + eps*noise 
    output = output.clamp(min=0,max=1)
        
    return output


def pretrain(epochs,model,train_loader, test_loader, criterion,optimizer_pre):
    embeddings_1 = []
    embeddings_2 =[]
    labels =[]
    for epoch in range(epochs):
        correct, num_example = 0,0
        losses = 0
        model.train()
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target= images.cuda(), target.cuda()
            out, m, rep_1, rep_2, rep_3 = model(images, test= False)
            loss_bce = criterion(out,target)
            loss = loss_bce 
            preds = out.data.max(1, keepdim=True)[1]
            correct += preds.eq(target.data.view_as(preds)).sum()
            num_example += len(target)
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f}  Accuracy:  {}".format(epoch+1,epochs, loss.item(), correct.item()/num_example)


            if batch_idx % p.log_interval == 0:
                print(to_print)
                
        tester.test_pre(model, test_loader)
        
        

        #if epoch % 10==0:
            #val_embeddings_1, val_embeddings_2, val_labels_baseline = extract_embeddings(test_loader, model,False)
            #embeddings_1.append(val_embeddings_1)
            #embeddings_2.append(val_embeddings_2)
            #labels.append(val_labels_baseline)
        
    #np.save("saved_model/embedding_1_pre", embeddings_1)
    #np.save("saved_model/embedding_2_pre", embeddings_2)
    #np.save("saved_model/labels_pre", labels)
    torch.save(model.state_dict(), "./saved_model/model_" + str("pretrain_model_mnist"))
    #plot_embeddings(val_embeddings_1, val_labels_baseline) 
    #plot_embeddings(val_embeddings_2, val_labels_baseline)    
    #attack_test(model, test_loader, nn.CrossEntropyLoss() )
    
    
def posttrain(epochs, model,train_loader, test_loader, criterions, optimizers,scheduler,cfg):
    embeddings_1 = []
    embeddings_2 =[]
    labels=[]
    criterion =  nn.CrossEntropyLoss() 
    br_64 = BarlowTwins(64).cuda()
    br_512 = BarlowTwins(512).cuda()
    br_1152 = BarlowTwins(1152).cuda()
    
    optimizer_br_64 = torch.optim.SGD(br_64.parameters(), lr=cfg['lr_br_64'])
    optimizer_br_512 = torch.optim.SGD(br_512.parameters(), lr=cfg['lr_br_512'])
    optimizer_br_1152 = torch.optim.SGD(br_1152.parameters(), lr=cfg['lr_br_1152'])
    
    lr_br_sc_64 = torch.optim.lr_scheduler.MultiStepLR(optimizer_br_64, milestones=[30,120,240,400], gamma=0.1)
    lr_br_sc_512 = torch.optim.lr_scheduler.MultiStepLR(optimizer_br_512, milestones=[30,120,240,400], gamma=0.1)
    lr_br_sc_1152= torch.optim.lr_scheduler.MultiStepLR(optimizer_br_1152, milestones=[30,120,240,400], gamma=0.1)
    
    for epoch in range(epochs):
        
        scheduler['lr_scheduler'].step()
        scheduler['lr_scheduler_64'].step()
        scheduler['lr_scheduler_512'].step()
        scheduler['lr_scheduler_1152'].step()
        scheduler['lr_conscheduler_64'].step()
        scheduler['lr_conscheduler_512'].step()
        scheduler['lr_conscheduler_1152'].step()
        lr_br_sc_64.step()
        lr_br_sc_512.step()
        lr_br_sc_1152.step()
        
        
        correct, num_example = 0,0
        model.train()
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target= images.cuda(), target.cuda()
            
            noise_images = add_noise(images)
            true_target= target
            data= torch.cat((images,noise_images),0)
            labels= torch.cat((target, true_target))
            
            out, m , rep_1, rep_2 = model(data,test=False)
            
            b = torch.split(rep_2, len(images), dim=0) # it returns a tuple
            b = list(b) # convert to list if you want
            a = torch.split(rep_1, len(images), dim=0) # it returns a tuple
            a = list(a) # convert to list if you want
            
            q = torch.split(m, len(images), dim=0) # it returns a tuple
            q = list(q) # convert to list if you want
            
            loss_br_64, rep2  = br_64(b[0],b[1])
            loss_br_512 , rep1   = br_512(a[0],a[1])
            loss_br_1152, m1 = br_1152(q[0],q[1])
            
            loss_bce = criterion(out,labels)
            loss_prox_512, _ = criterions["criterion_prox_512"](rep_1, labels) 
            loss_prox_64, preds = criterions["criterion_prox_64"](rep_2, labels) 
            loss_prox_1152, _ = criterions["criterion_prox_1152"](m, labels) 
            
            loss_conprox_512, _ = criterions["criterion_conprox_512"](rep_1, labels) 
            loss_conprox_64, _ = criterions["criterion_conprox_64"](rep_2, labels) 
            loss_conprox_1152, _ = criterions["criterion_conprox_1152"](m, labels)
            
            
            if epoch <= 20:
                loss = loss_bce  + (loss_prox_64 - p.we*loss_conprox_64)  + (loss_prox_512 - p.we*loss_conprox_512) + (loss_prox_1152 - p.we*loss_conprox_1152) + 0*loss_br_64 + 0*loss_br_512 + 0*loss_br_1152
                
            else:
                loss = loss_bce  + (loss_prox_64 - p.we*loss_conprox_64)  + (loss_prox_512 - p.we*loss_conprox_512) + (loss_prox_1152 - p.we*loss_conprox_1152) + 0.01*loss_br_64 + 0.01*loss_br_512 + 0.01*loss_br_1152
                
            
            
            correct += preds.eq(labels.data.view_as(preds)).sum()
            num_example += len(labels)
            optimizers["optimizer_post"].zero_grad()
            optimizers['optimizer_prox_512'].zero_grad() 
            optimizers['optimizer_prox_64'].zero_grad() 
            optimizers['optimizer_prox_1152'].zero_grad() 
            
            optimizers['optimizer_conprox_512'].zero_grad() 
            optimizers['optimizer_conprox_64'].zero_grad() 
            optimizers['optimizer_conprox_1152'].zero_grad() 
            
            optimizer_br_64.zero_grad()
            optimizer_br_512.zero_grad()
            optimizer_br_1152.zero_grad()
            
            loss.backward()
            optimizers['optimizer_post'].step()
            
            for param in criterions['criterion_prox_64'].parameters():
                param.grad.data *= (1. /1)
            optimizers['optimizer_prox_64'].step()


            for param in criterions['criterion_prox_512'].parameters():
                param.grad.data *= (1. /1)
            optimizers['optimizer_prox_512'].step()
            
            
            for param in criterions['criterion_prox_1152'].parameters():
                param.grad.data *= (1. /1)
            optimizers['optimizer_prox_1152'].step()
            
            
            for param in br_64.parameters():
                 param.grad.data *= (1. /0.01)
            optimizer_br_64.step()


            for param in br_512.parameters():
                param.grad.data *= (1. /0.01)
            optimizer_br_512.step()
            
            
            for param in br_1152.parameters():
                param.grad.data *= (1. /0.01)
            optimizer_br_1152.step()
            
            
            
            for param in criterions['criterion_conprox_64'].parameters():
                param.grad.data *= (1. /p.we)
            optimizers['optimizer_conprox_64'].step()


            for param in criterions['criterion_conprox_512'].parameters():
                param.grad.data *= (1. /p.we)
            optimizers['optimizer_conprox_512'].step()
            
            
            for param in criterions['criterion_conprox_1152'].parameters():
                param.grad.data *= (1. /p.we)
            optimizers['optimizer_conprox_1152'].step()
            
            to_print = "Epoch[{}/{}] Loss: {:.3f} loss_br_64 {:.3f} loss_br_512 {:.3f} loss_br_64 {:.3f}  Accuracy:  {}".format(epoch+1,epochs, loss.item(),loss_br_64.item() , loss_br_512.item(), loss_br_1152, correct.item()/num_example)


            if batch_idx % p.log_interval == 0:
                print(to_print)

        #helper.plot_histogram(epoch,idx, pre_wts, list(model.embedding_net.layers.parameters()), list(gmm.parameters()), correct.item()/num_example,"./figs/")    
        #test_loss, test_correct, test_instance_counter = tester.test(model, test_loader, criterions) 
        
        if epoch % 10==0:
        #val_embeddings_1, val_embeddings_2, val_labels_baseline = extract_embeddings(test_loader, model)
            #plot_embeddings(val_embeddings_1, val_labels_baseline) 
        #plot_embeddings(epoch,val_embeddings_2, val_labels_baseline)    
            #val_embeddings_1, val_embeddings_2, val_labels_baseline = extract_embeddings(test_loader, model,False)
            #embeddings_1.append(val_embeddings_1)
            #embeddings_2.append(val_embeddings_2)
            #labels.append(val_labels_baseline)
            
            test_loss, test_correct, test_instance_counter = tester.test(model, test_loader, criterions) 
        
    #np.save("saved_model/embedding_1_post", embeddings_1)
    #np.save("saved_model/embedding_2_post", embeddings_2)
    #np.save("saved_model/labels_post", labels)
    #torch.save(model.state_dict(), "./saved_model/model_" + str('posttrain_model'))
    torch.save(model.state_dict(), "./saved_model/model_" + str("posttrain_model_mnist_prox"))
    
    
    
    return test_correct / (test_instance_counter*1.0) * 100.0 

