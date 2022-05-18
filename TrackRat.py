''' Machine learning for tracking a rat '''

import pickle
import torch
import time
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch import nn
from torch_lr_finder import LRFinder

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Just so torch_lr_finder doesn't clash

### Define dataset
class Data(torch.utils.data.Dataset):
    def __init__(self, imgpath, labelpath):
        # Load images from the path and reorder so they are actually in order
        self.imgpath = imgpath
        self.images = os.listdir(self.imgpath)
        numbers = [int(n[5:len(n)-4]) for n in self.images]
        keys = np.argsort(numbers)
        self.images = [self.images[i] for i in keys]
        # Get labels from the labels file
        file = open(labelpath + '/Labels.pkl', 'rb')
        self.labels = pickle.load(file)
        self.labels = (torch.from_numpy(self.labels)).long()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Return correct image and label pair
        img_name = self.images[idx]
        img = cv2.imread(self.imgpath + '/' + img_name)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1) # Dimensions in order of channels, height, width
        img_tensor = (img_tensor/255).float() # Normalize and convert to float
        
        return img_tensor, self.labels[idx, :]
    
### Dataloader
imgpath = os.getcwd() + '/ChocolateImages'
labelpath = os.getcwd() + '~/ChocolateLabels'
data = Data(imgpath, labelpath)
batch_size = 32
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
loader= iter(dataloader)

### For defining height and width
img = cv2.imread(imgpath + '/Frame0.jpg')
height = img.shape[0]
width = img.shape[1]

### Define model
class positionmodel(nn.Module):                    
    def __init__(self, h, w): 
        super(positionmodel,self).__init__()    
        self.h = h
        self.w = w   
        self.resnet = torchvision.models.resnet50(pretrained = True)            
                                                
    def forward(self, x):     
        # Step through resnet until right before the fc layer                  
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        
        x = torch.squeeze(x)
                
        # Define a new fc layer for each coordinate
        self.fcx = nn.Sequential(             
            nn.Linear(x.shape[1], 1024),
            nn.ReLU(),
            
            nn.Linear(1024, self.w),
            nn.Softmax(dim=1),
        )  
        self.fcy = nn.Sequential(             
            nn.Linear(x.shape[1], 1024),
            nn.ReLU(),
            
            nn.Linear(1024, self.h),
            nn.Softmax(dim=1),
        )                                 
        
        xpos = self.fcx(x)
        ypos = self.fcy(x)
        return xpos, ypos, x

### Setup model, optimizer, and loss criterion
model = positionmodel(height, width)
model = model.float()
test = torch.ones((10, 3, height, width))
x_out = model(test)  # The model doesn't recognize everything untill this step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
criterion = nn.CrossEntropyLoss()

### Freeze pretrained model
for name, param in model.named_parameters():
    print(name)
    if 'fcx' not in name and 'fcy' not in name:
        param.requires_grad_(False) 

# ### Find LR, must modify model to return one output and dataloader to return one label
# lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
# lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
# lr_finder.plot()

### Mini-batch training
epochs = 10
TrainLoss = np.zeros((epochs, len(dataloader)))
for e in range(epochs):
    for (b, data) in enumerate(dataloader):
        start = time.time()
        img = data[0]
        label = data[1]
        print('Epoch:', e+1 , '/', epochs, ' Batch:', b+1, '/', len(dataloader))
        
        print('Forward pass...')
        x_out, y_out, _ = model(img)
                
        lossx = nn.functional.cross_entropy(x_out, label[:, 0])
        
        lossy = nn.functional.cross_entropy(y_out, label[:, 1])
        
        loss = (lossx+lossy)/2
        del lossx, lossy
                
        TrainLoss[e,b] = loss.item()
        
        print('Backward pass...')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Training Loss:', TrainLoss[e,b])
        print('Time:', time.time()-start)

    
## Save trained model
modelfile = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/Code/Chocolate1T_07_14_14_model.pt'
torch.save(model.state_dict(), modelfile)
print('Saved')

    
''' For debugging '''     
## Display two images, show the output of resnet and the x and y softmax layers
temp = img[0,:,:,:]
temp = temp.permute(1, 2, 0)
temp = temp.numpy()
cv2.imshow('1',temp)

temp2 = img[1,:,:,:]
temp2 = temp2.permute(1, 2, 0)
temp2 = temp2.numpy()
cv2.imshow('2',temp2)

x_out, y_out, x = model(img[0:1,:,:,:])    

plt.figure()
plt.plot(x.detach().numpy()[0,:])
plt.plot(x.detach().numpy()[1,:])

plt.figure()
plt.plot(x_out.detach().numpy()[0,:])
plt.plot(x_out.detach().numpy()[1,:])

plt.figure()
plt.plot(y_out.detach().numpy()[0,:])
plt.plot(y_out.detach().numpy()[1,:])





