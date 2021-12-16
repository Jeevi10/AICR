import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model import *  # Imports the ResNet Model
from networks import *
"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
num_classes=10
#torch.cuda.set_device(0)
#model = resnet(num_classes=num_classes,depth=110)
model = sixNet()
#mdoel = model.cuda()
if True:
    model = nn.DataParallel(model).cuda()
    
#Loading Trained Model
softmax_filename= './saved_model/model_pretrain_model_mnist'
#filename= 'Models_PCL/CIFAR10_PCL.pth.tar' 
robust_model=  './saved_model/model_posttrain_model_mnist_L'
checkpoint = torch.load(softmax_filename)
model.load_state_dict(checkpoint)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(),])
    
testset = torchvision.datasets.MNIST(root='./file', train=False,
                                         download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, pin_memory=True,
                                          shuffle=False, num_workers=4)

# Mean and Standard Deiation of the Dataset
mean = [0.1307]
std = [0.3081]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]

    return t
def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    return t

# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 20/ 255
    else:
        step = eps / iterations
        
        noise = 0
        
    for j in range(iterations):
        out_adv = model(normalize(adv.clone()),False)
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0, 1)

        adv.grad.data.zero_()

    return adv.detach()

# Loss Criteria
criterion = nn.CrossEntropyLoss()
adv_acc = 0
clean_acc = 0
eps = 80/255 # Epsilon for Adversarial Attack

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    
    clean_acc += torch.sum(model(normalize(img.clone().detach())).argmax(dim=-1) == label).item()
    adv= attack(model, criterion, img, label, eps=eps, attack_type= 'fgsm', iters= 10 )
    adv_acc += torch.sum(model(normalize(adv.clone().detach())).argmax(dim=-1) == label).item()
    print('Batch: {0}'.format(i))
print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / len(testset), adv_acc / len(testset)))