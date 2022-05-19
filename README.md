# TrackRatCNN 

*Work in progress*
 
As part of learning more about machine learning and neural networks I challenged myself to redo the rat tracking done in the TrackRat repository with a convolutional neural network rather than a heuristic. 

LabelData.py allows for manual labeling of each frame of a video and saves each frame as an image. TrackRat then uses the images and labels to create a PyTorch Dataset class for use in a Dataloader. 

I am currently using transfer learning with a pretrained network (ResNet) and re-formatting the fully connected layer into a linear output with ReLU activation for the two coordinates. The loss function is the root mean squared error, which essentially gives the distance of the correct pixel to the predicted pixel. I have configured the model to run on a GPU (CUDA) for speed (roughly 100 times faster).

## To do
- ~~Implement Dataset and Dataloader~~
- ~~Define model~~
- ~~Run model on GPU~~
- Determine if current version of the network can learn coordinates at all. 
  - The pretrained networks have a preprocessing step for resizing images. Rather than inputting the raw images directly to the network, build a CNN to reduce it first. 
    - Downsample first (2x should not affect image quality much)
    - Can build an autoencoder-esque preprocessing step, or just allow the weights in the initial CNN to be trained
  - Can try larger ResNet or deeper fully connected layer for higher accuracy
  - Batch normalization might help - worth testing
  - Hyperparameter tuning
- Re-save the video with correct label and the prediction overlaid for manual assessment
  - Large errors could be related to instances when the animal is rearing
- Split the train and test sets in the Dataset and automate cross validation
  - Use to inform need of regularization
- Compare different networks for transfer learning
  - ResNet18, ResNet50, Inceptionnet, etc.
- Test on completely different dataset
  - Iterate further tuning
