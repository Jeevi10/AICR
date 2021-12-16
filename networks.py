import torch.nn as nn
import torch.nn.functional as F
import parameter as p
from torch.autograd import Variable
import torch 

vis_size = 3

def w_relu(x):
    return F.relu(x) #- F.relu(x-1)*0.75

class LeNet(nn.Module):
    def __init__(self, pretrainin=False):
        super(LeNet, self).__init__()
        conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=1, bias=p.use_bias)
        conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1, bias=p.use_bias)
        conv3 = nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=p.use_bias)
        conv4 = nn.Conv2d(128, 256, kernel_size=3,padding=1, bias=p.use_bias)
        fc1 = nn.Linear(12544, 512 , bias=p.use_bias)
        fc2 = nn.Linear(512,64,bias=p.use_bias)
        
        layers = [conv1, conv2, conv3, conv4, fc1,fc2]
        self.layers = nn.ModuleList(layers)
        
        
        # embeddings
        self.emb_1 = nn.Linear(512, vis_size, bias=False)
        self.emb_2 = nn.Linear(64, vis_size, bias=False)
            
        
    def forward(self, x, class_index = -1,test=True):
        x = w_relu((self.layers[0](x)))
        x = w_relu(F.max_pool2d((self.layers[1](x)),2))

        x = w_relu((self.layers[2](x)))
        x = w_relu(F.max_pool2d((self.layers[3](x)),2))
        x = x.view(-1,12544)
        representation_1 = self.layers[4](x)
        representation_1_r = w_relu(representation_1)
        representation_2 = self.layers[5](representation_1_r)
        self.hidden = representation_1.detach().clone()
        #embedding_1 = self.emb_1(representation_1)
        #embedding_2 = self.emb_2(representation_2)
        
        return None, None, representation_1, representation_2
    
    
class ClassificationNet(nn.Module):
    def __init__(self, LeNet, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = LeNet
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(64, n_classes)

    def forward(self, x, test=True):
        embedding_1, embedding_2, representation_1, representation_2 = self.embedding_net(x)
        output = self.nonlinear(representation_2)
        scores = self.fc1(output)
        
        if test == True:
            return scores
        
        return None, representation_2, representation_1, scores

    def get_embedding(self, x, pretrain=True):
        embedding_1, embedding_2, representation_1, representation_2 = self.embedding_net(x)
        if pretrain==True:
            return embedding_1, embedding_2
        return self.nonlinear(embedding_1), self.nonlinear(embedding_2)
    
    
    
def mnist_net(images):
    inputs = tf.transpose(images, perm=[0,2,3,1])
    
    conv1 = Conv(inputs, [5,5,1,32], activation=ReLU)

    conv2 = Conv(conv1, [5,5,32,32], activation=ReLU)

    pool1 = Max_pool(conv2, padding='VALID')

    conv3 = Conv(pool1, [5,5,32,64], activation=ReLU)

    conv4 = Conv(conv3, [5,5,64,64], activation=ReLU)

    pool2 = Max_pool(conv4, padding='VALID')

    conv5 = Conv(pool2, [5,5,64,128], activation=ReLU)

    conv6 = Conv(conv5, [5,5,128,128], activation=ReLU)

    pool3 = Max_pool(conv6, padding='VALID')

    fc1 = FC(tf.reshape(pool3, [-1, 3*3*128]), 3*3*128, 2)
    fc1_out = ReLU(fc1)

    logits = FC(fc1_out, 2, 10)

    return fc1, logits



class sixNet(nn.Module):
    def __init__(self, pretrainin=False):
        super(sixNet, self).__init__()
        conv1 = nn.Conv2d(1, 32, kernel_size=5,padding=2, bias=p.use_bias)
        conv2 = nn.Conv2d(32, 32, kernel_size=5,padding=2, bias=p.use_bias)
        pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        conv3 = nn.Conv2d(32, 64, kernel_size=5,padding=2, bias=p.use_bias)
        conv4 = nn.Conv2d(64, 64, kernel_size=5,padding=2, bias=p.use_bias)
        pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        
        conv5 = nn.Conv2d(64, 128, kernel_size=5,padding=2, bias=p.use_bias)
        conv6 = nn.Conv2d(128, 128, kernel_size=5,padding=2, bias=p.use_bias)
        pool3 = nn.MaxPool2d(2, stride=2, padding=0)


        fc1 = nn.Linear(1152, 512 , bias=p.use_bias)
        fc2 = nn.Linear(512,64,bias=p.use_bias)
        fc_out = nn.Linear(64,p.n_classes,p.use_bias)
        layers = [conv1, conv2, conv3, conv4, conv5, conv6, fc1,fc2,fc_out]
        self.layers = nn.ModuleList(layers)
        self.pooling = nn.ModuleList([pool1,pool2,pool3])
            
        
    def forward(self, x, class_index = -1,test=True):
        x = w_relu((self.layers[0](x)))
        x = w_relu(self.pooling[0](self.layers[1](x)))
        x = w_relu((self.layers[2](x)))
        x = w_relu(self.pooling[1](self.layers[3](x)))
        x = w_relu((self.layers[4](x)))
        x = w_relu(self.pooling[2](self.layers[5](x)))
        
        m = x.view(-1,3*3*128)
        
        representation_1 = self.layers[6](m)
        #representation_1_r = w_relu(representation_1)
        representation_2 = self.layers[7](representation_1)
        #representation_2_r = w_relu(representation_2)
        
        logits = self.layers[-1](representation_2)
        
        if test == True:
            return logits
        
        
        return logits, m , representation_1, representation_2
    
    

