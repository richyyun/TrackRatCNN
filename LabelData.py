''' For labeling a video with x,y coordinates of the nose '''

import cv2
import numpy as np
import pickle

## Global variables for mouse positions
mouseX = 0
mouseY = 0

## Click callback function for opencv
def click(event, x, y, flags, params):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

## Video setup
vidname = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/Videos/Chocolate1T_07_14_14.mpg'
video = cv2.VideoCapture(vidname)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click)

## Loop through and get mouse positions on click of each frame
pos = np.zeros((length,2))
for f in range(length):
    _, frame = video.read()
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    
    pos[f, 0] = mouseX
    pos[f, 1] = mouseY
    
    print(f, mouseX, mouseY)
    
cv2.destroyAllWindows()    

## Save positions
savename = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/Videos/Chocolate1T_07_14_14.pkl'
# file = open(savename,'wb')
# pickle.dump(pos, file)
# file.close()

# To read
file = open(savename,'rb')
pos = pickle.load(file)
file.close()

## Write video with positions marked
video = cv2.VideoCapture(vidname)
labelvid = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/Videos/Chocolate1T_07_14_14_Labeled.avi'
out = cv2.VideoWriter(labelvid, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

# Set up data for pytorch
X = np.zeros((length, 1, height, width), dtype = np.ubyte)
for f in range(length):
    _, frame = video.read()
    X[f, 0, :, :] = frame[:, :, 0] # grayscale so just take one channel
    cv2.circle(frame, (int(pos[f, 0]), int(pos[f, 1])), 5, (0, 0, 255), thickness=2)
    if f%100 == 0:
        print(f)
    out.write(frame)
    
cv2.destroyAllWindows()    
out.release


## Save data and labels together
Y = pos
labeleddata = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/Videos/Chocolate1T_07_14_14_full.pkl'
file = open(labeleddata,'wb')
pickle.dump([X, Y], file)


## Save as individual images for PyTorch Dataloader
video = cv2.VideoCapture(vidname)
imfile = 'C:/Users/Richy Yun/Dropbox/Projects/RatTracking/ChocolateImages/Frame'
for f in range(length):
    _, frame = video.read()
    cv2.imwrite(imfile+str(f)+'.jpg', frame)
    
    
    
    
    
    
    
    