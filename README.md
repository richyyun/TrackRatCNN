# TrackRatCNN

As part of learning more about machine learning and neural networks I challenged myself to redo the rat tracking done in the TrackRat repository with a convolutional neural network rather than a heuristic. 

LabelData.py allows for manual labeling of each frame of a video and saves each frame as an image. TrackRat then uses the images and labels to create a PyTorch Dataset class for use in a Dataloader. 

I am currently using a pretrained network (resnet50) and re-formatting the fully connected layer into two parallel layers with a softmax output each for the x- and y-coordinates for the position of the rat's nose. 
