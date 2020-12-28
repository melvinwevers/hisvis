# Repository for HisVis

NWO-Kiem Project: 300-1-0060.
This project explores how computer vision technology can be adapted to a historical photo collection. We focus on one particular computer vision task: scene detection. This task tries to describe ``the place in which the objects seat'', rather than classifying the object depicted on an image, or locating particular object in a picture by drawing bounding boxes around them. Rather than merely detecting object, the ability to search for images based on the scene represented in them is a useful feature for heritage institutions, especially since this information is often not included in the meta data. Using this information, cultural historians could examine the representations of particular scenes, for example protests, at scale. More specifically, this project applies transfer learning to Places-365, a scene detection model.

As a starting point, we use a PyTorch ResNet-50 model---a convolutional neural network of fifty layers--- trained on the Places-365 dataset.\footnote{\url{https://github.com/CSAILVision/places365}} We load a pretrained-Places365 model, for which we unfreeze the final fully-connected layers, add a ReLU layer, a dropout layer, a LogSoftmax layer, and the AdamW optimizer.\footnote{We also experimented with different dropout values, and unfreezing of deeper layers}. To counter for the imbalance of the training categories, we experimented with weight-balancing the training data.\footnote{When weight balancing, we also slightly increased the dropout, as the model was more prone to overfitting in this case}

Without using weight balancing, we reached a mean top1-accuracy for 10-fold cross-validation is 50.29 with a considerable standard deviation of 24.25. The top5-accuracy is 79.70 with a standard deviation of 18.69.

When using weight balancing, 

## Data

processed: contains the training set. These images can be downloaded from Zenodo.

## Models

Contains the trained models. 

## Notebooks

Notebooks for exploring the training data and the results/

## Output

_Figures_ contains generated figures
_Results_ contains accuracy scores per class, cross_validation results, and predictions on the validation set.

## SSRC

`train_deBoer.py` - the script for transfer learning

Example usage:
`train_deBoer.py --shuffle --cross_val --feature_extract --balanced`
This trains the model with cross_validation on the final layers using a shuffled and balanced training set.

`helper.py` - helper functions

`resize.sh` script for batch resizing images

`make_split.py` constructing training set with optional test set

`make_predictions.py` making predictions on validation set which can be used to construct a confusion matrix.

