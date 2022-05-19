# TrackRatCNN 

*Work in progress*
 
As part of learning more about machine learning and neural networks I challenged myself to redo the rat tracking done in the TrackRat repository with a convolutional neural network rather than a heuristic. 

LabelData.py allows for manual labeling of each frame of a video and saves each frame as an image. TrackRat then uses the images and labels to create a PyTorch Dataset class for use in a Dataloader. 

I am currently using transfer learning with a pretrained network (ResNet) and re-formatting the fully connected layer into a linear output with ReLU activation for the two coordinates. The loss function is the root mean squared error, which essentially gives the distance of the correct pixel to the predicted pixel. I have configured the model to run on a GPU (CUDA) for speed (roughly 100 times faster).

To do:
- ~~Implement Dataset and Dataloader~~
- ~~Run model on GPU~~
- Determine if current version of the network can learn coordinates at all. 
 - May need to resize image to be smaller for more efficient training
 - May require more epochs
 - Hyperparameter tuning
 - Batch normalization might help 
- Re-save the video with correct label and the prediction overlaid for manual assessment
 - Errors could be related to instances when the animal is rearing
- Split the train and test sets in the Dataset and automate cross validation
 - Use to inform need of regularization
- Test on completely different dataset
 - Iterate further tuning
