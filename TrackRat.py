''' CNN for tracking a rat '''
import pickle
import torch
import time
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch import nn
# from torch_lr_finder import LRFinder

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Just so torch_lr_finder doesn't clash

# from GPUtil import showUtilization as gpu_usage # For debugging cuda
# from numba import cuda # Needed for emptying GPU cache

# # To clear GPU cache (may need to restart kernel after)
# gpu_usage()
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)
# gpu_usage()


''' Define device '''
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
   
    # Return correct image and label pair
    def __getitem__(self, idx):
        
        # Get image
        img_name = self.images[idx]
        img = cv2.imread(self.imgpath + '/' + img_name)
        
        # Transform image
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1) # Dimensions in order of channels, height, width
        img_tensor = img_tensor[0,:,:] # Black and white images from an IR camera so only need one channel
        img_tensor = img_tensor[None, :, :] # Previous step collapses channel dimension, so readd
        img_tensor = img_tensor.float()
        
        return img_tensor, self.labels[idx, :]
    
### Setup Dataloader
# Path to data
imgpath = os.getcwd() + '/ChocolateImages'
labelpath = os.getcwd() + '/ChocolateLabels'

# Define dataset
data = Data(imgpath, labelpath)
# Split dataset into train and test data
testsize = int(0.2*np.floor(len(data)))
trainsize = len(data) - testsize
train_data, test_data = torch.utils.data.random_split(data, [trainsize, testsize])
# Define dataloaders
batch_size = 32     # Seems to work best 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

''' Define model '''
class positionmodel(nn.Module):                    
    def __init__(self): 
        super(positionmodel,self).__init__()    
        
        # Make image smaller (halving the height and width)
        self.Resize= torchvision.transforms.Resize(240) 
        
        # Normalize image
        self.Normalize = torchvision.transforms.Normalize(mean = 0.5, std = 0.25)
        
        # Number of neurons for convolutional layers
        filters = 128
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size = (5,5), stride = (2, 2), padding = (2, 2)),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 5, stride = 2, padding = 2)
        )
        
        # Convolution layers
        self.BasicBlock = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.BatchNorm2d(filters)
        )         
        
        # Shrink parameter count so fully connected layers aren't too large
        self.AvgPool = nn.AdaptiveAvgPool2d((10, 10))
        
        self.fcin = filters*10*10
        
        # Has to be in init. If defined in forward, device not considered to be cuda
        # Fully connected layers, returning two values that correspond to x-y coordinates
        # Probably don't need as many layers?
        self.fc = nn.Sequential(             
            nn.Linear(self.fcin, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 2),
            nn.ReLU(),
        )          
                               
    def forward(self, x):     
        
        # Format image
        x = self.Resize(x)
        x = self.Normalize(x)
        
        # Initial convolution
        x = self.Conv1(x)
        
        # Convolution layers
        x = self.BasicBlock(x)
        x = self.BasicBlock(x)
        x = self.BasicBlock(x)
        x = self.BasicBlock(x)
        
        # Lower parameters and flatten to 1D array per input sample
        y = self.AvgPool(x)
        y = torch.flatten(y, start_dim=1)
        
        # Fully connected layers
        pos = self.fc(y)

        return pos, x, y # Returning x and y for debugging purposes

### Setup model, optimizer, and loss criterion
model = positionmodel()
model = model.float()
model = model.to(device)
# next(model.parameters()).device # For debugging
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)

# ### Find LR, must modify model to return one output and dataloader to return one 
# criterion = nn.MSELoss()
# lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
# lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
# lr_finder.plot()


''' Train model '''
### Mini-batch training
epochs = 100
TrainLoss = np.zeros((epochs, len(trainloader)))
TestLoss = np.zeros((epochs, len(testloader)))
verbose_steps = 10         # How many batches to wait before printing info
start = time.time()

for e in range(epochs):
    
    # Loop through train dataset
    b = 0                   # Batch number
    for img, label in trainloader:
        
        # Put on CUDA 
        img = img.to(device)
        label = label.to(device)
                
        # print('Forward pass...') # For debugging
        out, _ , _ = model(img)
                
        # Mean squared loss
        loss = nn.functional.mse_loss(out, label.float())
        
        TrainLoss[e,b] = loss.item()
        
        # print('Backward pass...') # For debugging
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if b%verbose_steps == 0:
            print('Epoch:', e+1 , '/', epochs, ' Batch:', b+1, '/', len(trainloader))
            print('Training Loss:', np.sqrt(TrainLoss[e,b]*2)) # Euclidean distance by pixels
            print('Time:', time.time()-start)
            
        b += 1
    
    # Loop through test dataset
    b = 0                   # Batch number
    for img, label in testloader:
        
        # Put on CUDA 
        img = img.to(device)
        label = label.to(device)
        
        # Get output of model        
        out, _ , _ = model(img)
                      
        # Mean squared loss
        loss = nn.functional.mse_loss(out, label.float())
        
        TestLoss[e,b] = loss.item()
        
        del loss    # Just to make sure the loss does not stay on the map even though item() should remove it
        
        if b%verbose_steps == 0:
            print('Epoch:', e+1 , '/', epochs, ' Batch:', b+1, '/', len(trainloader))
            print('Test Loss:', np.sqrt(TestLoss[e,b]*2)) # Euclidean distance by pixels
            print('Time:', time.time()-start)
            
        b += 1


''' Save '''
## Save trained model
modelfile = os.getcwd() + '/Models/Custom_TrainTest.pt'
torch.save(model.state_dict(), modelfile)
print('Model Saved')

# # To load
# model = positionmodel()
# model.load_state_dict(torch.load(modelfile))

## Save losses
lossfile = os.getcwd() + '/Losses/Custom_TrainTest.pkl'
file = open(lossfile,'wb')
pickle.dump([TrainLoss, TestLoss], file)
print('Losses Saved')
file.close()

# # To load
# file = open(lossfile,'rb')
# TrainLoss, TestLoss = pickle.load(file)
# file.close()


''' Plot Losses '''
# Each batch per epoch
plt.figure()
plt.plot(TrainLoss.T)
# Average per epoch
plt.figure()
plt.plot(np.mean(TrainLoss, axis=1))
plt.yscale('log')

# Difference per epoch
avg = np.mean(TrainLoss, axis=1)
diff = [avg[i] - avg[i-1] for i in range(1,len(avg))]
plt.figure()
plt.plot(diff)


''' Load and re-save video with true positions and prediction '''
vidname = os.getcwd() + '/Videos/Chocolate1T_07_14_14.mpg'
video = cv2.VideoCapture(vidname)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
video = cv2.VideoCapture(vidname)
labelvid = os.getcwd() + '/Videos/Chocolate1T_07_14_14_Predict_Custom.avi'
out = cv2.VideoWriter(labelvid, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

# Print video. Probably a faster way to do this, but sufficiently fast for now
for f in range(length):
    
    _, frame = video.read()
    temp = data.__getitem__(f)
    pos = temp[1].numpy()
    # Red is "true" location
    cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), thickness=2)
    
    img = temp[0]
    img = img.to(device)
    pred, _, _ = model(img[None, :])
    pred = pred.cpu().detach().numpy().squeeze()
    # Blue is prediction
    cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, (255, 0, 0), thickness=2) 
    
    if f%100 == 0:
        print(f)
    out.write(frame)
    
out.release






