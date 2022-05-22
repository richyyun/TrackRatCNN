# TrackRatCNN 

*Work in progress*
 
As part of learning more about machine learning and neural networks I challenged myself to redo the rat tracking done in the TrackRat repository with a convolutional neural network rather than a heuristic. This is purely a personal project and not part of my research.

LabelData.py allows for manual labeling of each frame of a video and saves each frame as an image. TrackRat then uses the images and labels to create a PyTorch Dataset class for use in a Dataloader. 

I initially tried using a pre-trained model (ResNet) for transfer learning, but it was not very accurate regardless of if I allowed all layers to train or not, likely due to the differences in input images (small color images vs wide greyscale) and the outputs (classification vs x-y coordinates). As a result I implemented my own network, shown in the current version of the code, which can be trained to detect the nose within roughly 10 pixel accuracy. I am currently tuning some hyperparameters of the network, including the number of layers and number of neurons (filters) per convolutional layer, and will move onto splitting the dataset into training and test sets to determine the need for regularization. 

Current diagram of the model:
<p align="center">
  <img src="https://github.com/richyyun/TrackRatCNN/blob/main/Diagram_20220522.png" />
</p>

## To do
1. ~~Implement Dataset and Dataloader~~
2. ~~Define model~~
3. ~~Run model on GPU~~
4. ~~Use transfer learning (ResNet or InceptionNet) with changes to the fully connected layers~~
5. Design a custom network 
   - Hyperparameter tuning **&larr; Currently in progress**
   - Re-save the video with correct label and the prediction overlaid for manual assessment
      - Large errors could be related to instances when the animal is rearing
6. Split the train and test sets in the Dataset and automate cross validation
  a. Use to inform need of regularization
7. Test on completely different dataset
  a. Iterate for further tuning
