import dataloader as dl
import torch
import parameter as p
from pathlib import Path



def create_exp_folder(model_name, decoder_name, class_loss_weight, k, remove_lastlayer, pushing):
    foldername = "model:%s_decoder:%s_classweight:%s_cluster:%s_remover_lastlayer:%s_mean_pushing:%s/" % (str(model_name), 
                                                                                                      str(decoder_name), 
                                                                                                      str(class_loss_weight), 
                                                                                                      str(k), str(remove_lastlayer),
                                                                                                      str(pushing))
    Path(foldername).mkdir(parents=True, exist_ok=True)
    return foldername



def get_dataloaders(model_name,dataset):
    # initialize the dataloaders
    train_loader = dl.get_trainloader(dataset)
    test_loader = dl.get_testloader(dataset)
    
    # binary classification training loaders for class-tie
    if model_name not in p.non_binary_models:
        loader_list = [] # loader_list[0] is the loader for class: 0 ...
        for i in range(p.n_classes):
            loader_list.append(dl.get_weighted_trainloader(i,dataset))
            
    return train_loader, test_loader, loader_list



def load_pretrained_models(network, decoder, model_name):
    print("loading the pretrained model..........")
    network.load_state_dict(torch.load("./saved_model/model_" + str(model_name)))
    decoder.load_state_dict(torch.load("./saved_model/dec_" + str(model_name)))
    decoder.eval()
    return network, decoder


def load_posttrained_models(network, reg, model_name, foldername):
    print("loading the posttrained model..........")
    network.load_state_dict(torch.load("./saved_model/"+foldername+"post_model_" + str(model_name)))
    reg.load_state_dict(torch.load("./saved_model/"+foldername+"post_gmm_" + str(model_name)))
    return network, reg