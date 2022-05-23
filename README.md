# TrackRatCNN 

\**Work in progress*\*
 
As part of learning more about machine learning and neural networks I challenged myself to redo the rat tracking done in the TrackRat repository with a convolutional neural network rather than a traditional computer vision heuristic. The video consists of a bottom-up view of a rat in a cylindrical chamber. I am also using Git for actual version control to refresh myself on the process. 

This is purely a personal project and not part of my research.

LabelData.py allows for manual labeling of each frame of a video and saves each frame as an image. TrackRat then uses the images and labels to create a PyTorch Dataset class for use in a Dataloader. 

I initially tried using a pre-trained model (ResNet) for transfer learning, but it was not very accurate regardless of if I allowed all layers to train or not, likely due to the differences in input images (small color images vs wide greyscale) and the outputs (classification vs x-y coordinates). As a result I implemented my own network, shown in the current version of the code, which can be trained to detect the nose within roughly 10 pixel accuracy. I am currently tuning some hyperparameters of the network, including the number of layers and number of neurons (filters) per convolutional layer, and will move onto splitting the dataset into training and test sets to determine the need for regularization. 

## To do
1. ~~Implement Dataset and Dataloader~~
2. ~~Define model~~
3. ~~Run model on GPU~~
4. ~~Use transfer learning (ResNet or InceptionNet) with changes to the fully connected layers~~
5. ~~Design a custom network
   - ~~Hyperparameter tuning
   - ~~Re-save the video with correct label and the prediction overlaid for manual assessment~~
     - ~~Large errors could be related to instances when the animal is rearing~~
6. Split the train and test sets in the Dataset and automate cross validation **&larr; Currently in progress**
  a. Use to inform need of regularization (already using L2 and getting some from batch norm)
7. Clean up code - split into other files
7. Test on completely different dataset
  a. Iterate for further tuning

## Work done so far
Diagram of the current model:
<p align="center">
  <img src="https://github.com/richyyun/TrackRatCNN/blob/main/Images/Diagram_20220522.png" />
</p>

MSE loss for each epoch when trained on the entire dataset:
<p align="center">
  <img width="500" src="https://github.com/richyyun/TrackRatCNN/blob/main/Images/Losses.png" />
</p>

34.6 in MSE translates to ~8.3 pixels in Euclidean distance from the true position to the predicted position. Although the error seems to still be decreasing, the changes were negligible (notice the logarithmic y-axis scale) so I chose to keep to 100 epochs for the time being to save time.

To ensure the model is doing well in all conditions (i.e. that it is consistent when the animal is rearing, using a nose-poke port for water, or grooming) I plotted the true and predicted positions on the original video. Below shows, from left to right, examples of the true position (red circle) and predicted position (blue circle) when the animal is rearing, using the nose poke, and grooming.

MSE loss for each epoch when trained on the entire dataset:
<p align="center">
  <img src="https://github.com/richyyun/TrackRatCNN/blob/main/Images/Predictions.png" />
</p>

## Lessons learned
- Using Dataset and Dataloader from PyTorch to setup the data. Splitting into train and test sets is very simple.
- Running the model on the GPU
  - All layers in the neural network model must be defined in the initializer to ensure all parameters are moved to CUDA. 
  - It is best to split up the training and testing sections into seperate methods. Otherwise there may be holdover on tensors in previous steps leading to OOM errors. 
  - Running on CUDA requires proper memory clearing, especially if  the code was interrupted part way through.
- Using pretrained networks is not necessarily the best approach. 
- Designing a custom NN
  - Batch normalization layers (and apparently Dropout layers as well) cannot be used multiple times, otherwise the averaging gets ruined when running in eval mode. 
