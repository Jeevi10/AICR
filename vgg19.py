import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import torch
import torch.nn as nn
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EPOCH = 50
N_CLASSES = 25


def read_mnist(train):
      return torchvision.datasets.MNIST( "./file", 
                                 train=train, 
                                 download=True, 
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,)),
                                 ]))


trainData = read_mnist(True)
testData = read_mnist(False)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

vgg16 = models.vgg16(pretrained=True)
      
#print(vgg16.classifier[6].out_features) # 1000 


# Freeze training for all layers
for i,param in enumerate(vgg16.features.parameters()):
    if i < 23:
        param.require_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 10)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
#print(vgg16)
vgg16.cuda()

# Loss, Optimizer & Scheduler
cost = tnn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    correct = 0
    total = 0
    for idx, (images, labels) in enumerate(trainLoader):
        images=images.repeat(1,3,1,1)
        images = images.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = cost(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        avg_loss += loss.data
        cnt += 1
        if idx% 100 ==0:
            print("[E: %d] loss: %f, avg_loss: %f,  Train_accu: %f" % (epoch, loss.data, avg_loss/cnt,100* correct.item()/total))
            
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

# Test the model
vgg16.eval()
correct = 0
total = 0

for idx,(images, labels) in enumerate(testLoader):
    images=images.repeat(1,3,1,1)
    
    images = images.cuda()
    outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100* correct.item()/total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'cnn.pkl')