import torch
import torchvision
import parameter as p
import numpy as np
import torchvision.transforms as transforms

def read_mnist(train):
      return torchvision.datasets.MNIST( "./file", 
                                 train=train, 
                                 download=True, 
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                                 ]))

    
def read_cifar(train):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./file/cifar10', train=True,
                                             download=True, transform=transform_train)
    
    if train==True:
        return trainset
    
    else:
        return torchvision.datasets.CIFAR10(root='./file/cifar10', train=train,
                                             download=True, transform=transform_test)
    
    
def read_fmnist(train):
    transform_train = transforms.Compose([
    #transforms.Pad(padding=4,fill=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
        ])
    
    transform_test = transforms.Compose([
        #transforms.Pad(padding=4,fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    
    trainset = torchvision.datasets.FashionMNIST(root='./file/fmnist', train=True,
                                             download=True, transform=transform_train)
    
    if train==True:
        return trainset
    
    else:
        return torchvision.datasets.FashionMNIST(root='./file/fmnist', train=train,
                                             download=True, transform=transform_test)
    
    

def my_collate(batch):
    """
    Reads p.n_classes to keep a subset of classes in the dataset.
    Used to subsample classes to have easier analysis of results.
    Gives error if the sampled classes do not start from the first class 
    (e.g., wont let you sample classes 2-5, has to start from 0).
    
    Parameters
    ----------
    batch: torch.tensor
        Batch of data to be collated.
    
    Returns
    -------
    dataloader.default_collate
    
    """
    modified_batch = []
    for item in batch:
        image, label = item
        class_list = [*range(0, p.n_classes, 1)] 
        if label in class_list:
            modified_batch.append(item)
    return torch.utils.data.dataloader.default_collate(modified_batch)


def get_weights(selected_class, dataset):
    """
    Used to get sampling weights for each instance in the dataset.
    Given the selected_class, makes sure that half of the batch contains
    the selected_class, and rest is equally sampled from rest of the classes.
    
    
    Parameters
    ----------
    selected_class: int
        Represents the index of the selected class.
    
    Returns
    -------
    all_weights: list
        List of sampling weights for each instance in the dataset.
    
    """
    if dataset == 'mnist':
        
        dataset_ = read_mnist(train=True)
        targets = dataset_.targets.detach().numpy()    
        
    elif dataset == 'cifar10':
        dataset_ = read_cifar(train=True)
        targets = np.array(dataset_.targets)
    elif dataset == 'fmnist':
        dataset_ = read_fmnist(train=True)
        targets = dataset_.targets.detach().numpy()  
        
     
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in targets])
    selected_class_weight = 0.5 / class_sample_count[selected_class] 
    other_class_diviser = 0.5 / ((p.num_classes)-1.0)
    all_weights = []

    for i in range(len(targets)):
        if targets[i] == selected_class:
            all_weights.append(selected_class_weight)
        else:
            num_instances = class_sample_count[targets[i]]
            all_weights.append(other_class_diviser / num_instances)
            
    return all_weights


def get_trainloader(dataset):
    """
    Returns a trainloader that contains MNIST training instances with given
    p.batch_size_train, and the collate function.
    Used to train the model for the main classification task.

    
    Parameters
    ----------
    
    Returns
    -------
    train_loader: torch.dataloader
        List of sampling weights for each instance in the dataset.
    
    """
    if dataset == 'mnist':
        
        dataset_ = read_mnist(train=True)
        
    elif dataset == 'cifar10':
        dataset_ = read_cifar(train=True)
        
    elif dataset == 'fmnist':
         dataset_ = read_fmnist(train=True)
        
    train_loader = torch.utils.data.DataLoader(dataset_, 
                                               batch_size=p.batch_size_train, 
                                               shuffle=True, 
                                               collate_fn=my_collate)
    return train_loader


def get_testloader(dataset):
    """
    Returns a test that contains MNIST test instances with given
    p.batch_size_test, and the collate function.
    Used to train the model for the main classification task.

    
    Parameters
    ----------
    
    Returns
    -------
    test_loader: torch.dataloader
        List of sampling weights for each instance in the dataset.
    
    """
    
    if dataset == 'mnist':
        
        dataset_ = read_mnist(train=False)
        
    elif dataset == 'cifar10':
        dataset_ = read_cifar(train=False)
        
    elif dataset == 'fmnist':
         dataset_ = read_fmnist(train=False)

    test_loader = torch.utils.data.DataLoader(dataset_, 
                                              batch_size=p.batch_size_test, 
                                              shuffle=True, 
                                              collate_fn=my_collate)
    
    return test_loader


def get_weighted_trainloader(class_index, dataset):
    """
    Given a class index, returns a trainloader that uses the
    sampling weights from get_weights to ensure class_index samples
    are half of the batch, and the other half equally contains rest
    of the classes.
    This function is used for the binary classification task to tie
    the classes to their corresponding clusters.

    
    Parameters
    ----------
    
    Returns
    -------
    test_loader: torch.dataloader
        List of sampling weights for each instance in the dataset.
    
    """
    
    if dataset == 'mnist':
        
        dataset_ = read_mnist(train=True)
        
    elif dataset == 'cifar10':
        dataset_ = read_cifar(train=True)
        
    elif dataset == 'fmnist':
         dataset_ = read_fmnist(train=True)


    sampler = torch.utils.data.sampler.WeightedRandomSampler(get_weights(class_index, dataset),
                                                             p.batch_size_class,
                                                             replacement=True)
    
    train_loader = torch.utils.data.DataLoader(dataset_, 
                                               batch_size=p.batch_size_class, 
                                               shuffle=False, sampler=sampler,
                                               collate_fn=my_collate)
    
    return train_loader
    
    
    
    