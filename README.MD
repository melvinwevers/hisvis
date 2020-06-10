# Repository for HisVis

NWO-Kiem Project: 300-1-0060

## data

processed: contains the training set. These images can be downloaded from Zenodo.

## models

Contains the trained models

## Notebooks

Notebooks for exploring the training data and the results

## Output

_Figures_ contains generated figures
_Results_ contains accuracy scores per class, cross_validation results, and predictions on the validation set. 

## SRC

`train_deBoer.py` - the script for transfer learning

Example usage:
`train_deBoer.py --shuffle --cross_val --feature_extract --balanced`
This trains the model with cross_validation on the final layers using a shuffled and balanced training set. 

`helper.py` - helper functions

`resize.sh` script for batch resizing images

`make_split.py` constructing training set with optional test set

`make_predictions.py` making predictions on validation set which can be used to construct a confusion matrix.

