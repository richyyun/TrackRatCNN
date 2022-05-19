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
from GPUtil import showUtilization as gpu_usage # For debugging cuda

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Just so torch_lr_finder doesn't clash

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


''' Dataset and Dataloader setup '''
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
labelpath = os.getcwd() + '/ChocolateLabels'
data = Data(imgpath, labelpath)
batch_size = 32
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
loader= iter(dataloader)

# ### For defining height and width
# img = cv2.imread(imgpath + '/Frame0.jpg')
# height = img.shape[0]
# width = img.shape[1]



''' Define model '''
class positionmodel(nn.Module):                    
    def __init__(self): 
        super(positionmodel,self).__init__()    
        # self.h = h
        # self.w = w   
        self.resnet = torchvision.models.resnet18(pretrained = True) # Pretrained ResNet            
                
        # Has to be in init. If defined in forward, device not considered to be cuda
        self.fc = nn.Sequential(             
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 2),
            nn.ReLU(),
        )          
                               
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
        
        # Apply our new fc layer                
        pos = self.fc(x)

        return pos, x #returning x for debugging purposes

### Setup model, optimizer, and loss criterion
model = positionmodel()
model = model.float()
model = model.to(device)
# next(model.parameters()).device # For debugging
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

### Freeze pretrained model
for name, param in model.named_parameters():
    print(name)
    # if 'fcx' not in name and 'fcy' not in name:
    if 'fc' not in name:
        param.requires_grad_(False) 

# ### Find LR, must modify model to return one output and dataloader to return one label
# lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
# lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
# lr_finder.plot()

''' Train model '''
### Mini-batch training
epochs = 100
TrainLoss = np.zeros((epochs, len(dataloader)))
for e in range(epochs):
    b = 0
    for img, label in dataloader:
        start = time.time()
        img = img.to(device)
        label = label.to(device)

        print('Epoch:', e+1 , '/', epochs, ' Batch:', b+1, '/', len(dataloader))
        
        print('Forward pass...')
        out, _ = model(img)
                
        # Root mean squared loss
        loss = torch.sqrt(nn.functional.mse_loss(out, label.float()))
        # loss = nn.functional.cross_entropy(out, label)
        
        TrainLoss[e,b] = loss.item()
        
        print('Backward pass...')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Training Loss:', TrainLoss[e,b])
        print('Time:', time.time()-start)
        b += 1

    
## Save trained model
modelfile = os.getcwd() + 'Models/Chocolate1T_07_14_14_model.pt'
torch.save(model.state_dict(), modelfile)
print('Saved')

    
''' For debugging '''     
## Display two images, show the output of ResNet and the final layer
temp = img[0,:,:,:]
temp = temp.permute(1, 2, 0)
temp = temp.numpy()
cv2.imshow('1',temp)

temp2 = img[1,:,:,:]
temp2 = temp2.permute(1, 2, 0)
temp2 = temp2.numpy()
cv2.imshow('2',temp2)

pos, x = model(img[0:1,:,:,:])    

plt.figure()
plt.plot(x.detach().numpy()[0,:])
plt.plot(x.detach().numpy()[1,:])

plt.figure()
plt.plot(pos[:,0].detach().numpy()[0,:])
plt.plot(pos[:,0].detach().numpy()[1,:])

plt.figure()
plt.plot(pos[:,1].detach().numpy()[0,:])
plt.plot(pos[:,1].detach().numpy()[1,:])





