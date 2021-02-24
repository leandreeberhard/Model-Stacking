# Model Stacking
This repository contains a demonstration of stacking three different models for more accurate prediction. 

The data used for this demo is from the [Kaggle Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge). 

The code is self-contained within the file `complete_pipeline.py` and can be run directly by placing it in a folder together with `train.csv` and `test.csv`. 


# Description

The three models that are stacked together are

* A 3-layer neural network with 100, 1000 and 1 layers, respectively. Additionally, there is a dropout with dropout probability 0.3 between each layer to avoid overfitting
* An XGBoost classifier with parameters `max_depth = 6` and `eta = 0.8`
* A KNN classifier using 10 nearest neighbors, with the influence of each neighbor on the final prediction weighted by the distance from the point to be predicted 

A so-called "meta-leaner" then takes the predictions from these three models and uses these as predictors to make a final prediction. Here, a simple logistic regression with L2 regularization is used as the meta-learner. 

To train the ensemble model, the training data set is first divided up into 10 disjoint folds. Then, the three models, which we train on the data not in the fold, make a prediction for the data points in the fold. We assemble all of these predictions into a new matrix with three columns and one prediction for each point in the training set from the three individual models. Finally, the meta-learner is trained on this new matrix.

It is important to train the individual models using the disjoint fold approach to ensure that the model generalizes properly. 
