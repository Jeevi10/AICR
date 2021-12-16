import torch
import printers
import torch.nn.functional as F

def test_pre(model, test_loader):
    """
    Test a given network using the data from test_loader.
    Print test statistics.
    
    Parameters
    ----------
    network: torch.nn
    test_loader: torch.dataloader
    
    Returns
    -------
    """
    
    test_losses = []    
    test_loss = 0
    correct = 0
    instance_counter = 0
    
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            instance_counter += len(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    # print results
    printers.test_print(test_loss, correct, instance_counter)
    
    return correct, instance_counter


def test(model, test_loader, criterions):
    """
    Test a given network using the data from test_loader.
    Print test statistics.
    
    Parameters
    ----------
    network: torch.nn
    test_loader: torch.dataloader
    
    Returns
    -------
    """
    
    test_losses = []    
    test_loss = 0
    correct = 0
    instance_counter = 0
    
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            instance_counter += len(target)
            out, m , rep_1, rep_2 = model(data,test=False)
            test_loss += F.cross_entropy(out, target, size_average=False).item()+ criterions["criterion_prox_512"](rep_1, target)[0].item() + criterions["criterion_prox_64"](rep_2, target)[0].item()+ criterions["criterion_prox_1152"](m, target)[0].item()
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    # print results
    printers.test_print(test_loss, correct, instance_counter)
    
    return test_loss, correct, instance_counter



def post_test(model, test_loader, criterion):
    test_losses = []    
    test_loss = 0
    correct = 0
    instance_counter = 0
    
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            instance_counter += len(target)
            _,_, _, rep2 = model(data, test=False)
            loss, preds = criterion(rep2, target)
            correct += preds.eq(target.data.view_as(preds)).sum()
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    # print results
    printers.test_print(test_loss, correct, instance_counter)
    
    return correct, instance_counter

