import numpy as np
import pandas as pd
from math import ceil, floor

print('Importing data...')
# import train data; delete ROLE_CODE since this corresponds exactly
train_data = pd.read_csv('train.csv')
train_labels = train_data['ACTION']
del train_data['ACTION']
del train_data['ROLE_CODE']

# import test data; delete ROLE_CODE
test_data = pd.read_csv('test.csv')
test_id = test_data['id']
del test_data['id']
del test_data['ROLE_CODE']

####################################
# split train data into folds
####################################
n_folds = 10
fold_size = ceil(train_data.shape[0] / n_folds)

# initialize remaining indices and fold dictionary
indices = [i for i in range(train_data.shape[0])]
folds = {}

# set seed for reproducibility
np.random.seed(10)

# shuffle indices
np.random.shuffle(indices)

# set first k-1 folds
for k in range(n_folds - 1):
    folds[k] = indices[fold_size * k:fold_size * (k + 1)]

# set last fold
folds[n_folds - 1] = indices[fold_size * (n_folds - 1):]


####################################
# process data for NN and XGB models
# NN and XGB use train_processed for training and test_processed for prediction
# KNN uses original train_data and test_data
####################################
print('Processing data...')

# convert data to categorical and consolidate rare feature levels into a single category
def convert_data_to_categorical(X_train_raw):
    X_train_processed = X_train_raw.copy(deep=True)

    for col in range(0, X_train_processed.shape[1]):
        levels = X_train_processed.iloc[:, col].value_counts().index.values

        vals_to_change = [level for level in levels if X_train_processed.iloc[:, col].value_counts()[level] <= 20]
        print(f'Processing column {col}...')
        n_rows = (X_train_processed.iloc[:, col]).shape[0]
        for row in range(n_rows):
            if X_train_processed.iat[row, col] in vals_to_change:
                X_train_processed.iat[row, col] = 0

    # encode features as categorical variables
    train_categorical = pd.get_dummies(X_train_processed, columns=X_train_processed.columns)

    return train_categorical


# augment training data to correct for class imbalance
def augment_training_data(X_train, y_train):
    n_maj, n_min = list(pd.DataFrame(y_train).value_counts())
    # number of times to replicate each minority class member
    n_repeat = floor(n_maj / n_min)

    # minority class indices (indices with 0)
    min_ind = np.equal(y_train, 0)
    X_train_min = X_train[min_ind]
    X_train_aug = np.concatenate([X_train] + [X_train_min] * n_repeat)

    y_train_min = y_train[min_ind]
    y_train_aug = np.concatenate([y_train] + [y_train_min] * n_repeat)

    # shuffle the indices
    perm = np.random.permutation(range(X_train_aug.shape[0]))
    X_train_aug = X_train_aug[perm]
    y_train_aug = y_train_aug[perm]

    return X_train_aug, y_train_aug


# combine train and test data into a single data frame for processing
train_test_data = pd.concat([train_data, test_data], ignore_index=True)

# process the combined data set
train_test_processed = convert_data_to_categorical(train_test_data)

# split back into two separate data frames
train_processed = train_test_processed[:train_data.shape[0]]
test_processed = train_test_processed[train_data.shape[0]:]

assert train_processed.shape[0] == train_data.shape[0] and test_processed.shape[0] == test_data.shape[0]


####################################
# NN model
####################################
import tensorflow as tf

# suppress warning messages
tf.logging.set_verbosity(tf.logging.ERROR)


# train neural network on augmented data
# n_repeat is the number of times to repeat each minority class point to match the majority class
def fit_nn(X_train_aug, y_train_aug, n_repeat=16):
    nn_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=(X_train_aug.shape[1],), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='Adam', loss='binary_crossentropy', weighted_metrics=['acc'])

    nn_model.fit(X_train_aug, y_train_aug, batch_size=256, epochs=10, verbose=0, validation_split=0.2,
              class_weight={1: (n_repeat + 1) / (n_repeat + 2), 0: 1 / (n_repeat + 2)})

    # return fitted model
    return nn_model


# make predictions from fitted NN model
def pred_nn(nn_fitted, X_test):
    # return predicted probability of class 1
    return nn_fitted.predict(X_test).reshape((X_test.shape[0],))


####################################
# XGBoost model
# xgb uses train_processed and augmented data
####################################
import xgboost as xgb


# train xgb on augmented data
def fit_xgb(X_train_aug, y_train_aug):
    # convert data into the correct format
    xgb_train = xgb.DMatrix(pd.DataFrame(X_train_aug).values, label=y_train_aug)

    # specify parameters
    params = {'max_depth': 6, 'eta': 0.8, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    num_round = 75

    # fit xgb model
    return xgb.train(params, xgb_train, num_round)


# make predictions from fitted model
def pred_xgb(xgb_fitted, X_test):
    # convert data into the correct format
    xgb_test = xgb.DMatrix(X_test.values)

    # make prediction
    return xgb_fitted.predict(xgb_test)


####################################
# KNN model
# knn uses the raw train_data
####################################
from sklearn.neighbors import KNeighborsClassifier


# fit a KNN model
def fit_knn(X_train, y_train):
    # specify model parameters
    knn_fitter = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='hamming', n_jobs=-1)

    # fit the model
    knn_fitter.fit(X_train, y_train)

    # return fitted model
    return knn_fitter


# make predictions using fitted model
def pred_knn(knn_fitted, X_test):
    return knn_fitted.predict_proba(X_test)[:, 1]



####################################
# generate predictions from all three models
####################################
print(f'Training models on {n_folds} folds...')

# initialize the data frame to store predictions from each model
level_one_data = pd.DataFrame(columns=['ind', 'nn', 'xgb', 'knn'])

# make predictions for all three models on each fold
for fold in folds.keys():
    print(f'Fold {fold+1} of {n_folds}...')
    raw_ind = sorted(folds[fold])
    raw_ind_set = set(raw_ind)

    train_ind = [i not in raw_ind_set for i in range(train_data.shape[0])]
    test_ind = [i in raw_ind_set for i in range(train_data.shape[0])]

    # augment training data for nn and xgb
    X_train_processed = train_processed[train_ind]
    X_test_processed = train_processed[test_ind]
    y_train = train_labels[train_ind]
    y_test = train_labels[test_ind]

    # augment the data for class balance for nn and xgb
    X_train_aug, y_train_aug = augment_training_data(X_train_processed, y_train)

    # data for knn
    X_train = train_data[train_ind]
    X_test = train_data[test_ind]

    # fit the three models on the train data for this fold
    print('Fitting NN...')
    n_maj, n_min = list(pd.DataFrame(y_train).value_counts())
    n_repeat = floor(n_maj / n_min)

    nn_fitted = fit_nn(X_train_aug, y_train_aug, n_repeat)
    print('Fitting XGB...')
    xgb_fitted = fit_xgb(X_train_aug, y_train_aug)
    print('Fitting KNN...')
    knn_fitted = fit_knn(X_train, y_train)

    # make predictions using these models
    print('Generating Predictions NN...')
    nn_preds = pred_nn(nn_fitted, X_test_processed)
    print('Generating Predictions XGB...')
    xgb_preds = pred_xgb(xgb_fitted, X_test_processed)
    print('Generating Predictions KNN...')
    knn_preds = pred_knn(knn_fitted, X_test)

    # assemble predictions into a DataFrame
    fold_preds = pd.DataFrame({'ind': raw_ind, 'nn': nn_preds, 'xgb': xgb_preds, 'knn': knn_preds})

    # merge with the full DataFrame
    level_one_data = pd.concat([level_one_data, fold_preds], ignore_index=True, axis=0)


# sort level_one_data back into the original order
level_one_data.sort_values(by='ind', axis=0, inplace=True)

# redefine index
level_one_data.index = level_one_data['ind']

# delete extra index column
del level_one_data['ind']


####################################
# train the meta learner on level_one_data
# choose logistic regression model for meta learner
####################################
print('Training meta learner...')
# logistic regression model for the meta learner
from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression(penalty='l2', C=1, class_weight='balanced', n_jobs=-1)
# fit the logistic regression
meta_model.fit(level_one_data, train_labels)

# model coefficients (it seems like knn has the largest effect on predictions)
meta_model.coef_


####################################
# train three original models on the full set of training data and make predictions for test data
####################################
# augment training data for NN and XGB
train_data_aug, train_labels_aug = augment_training_data(train_processed, train_labels)

print('Fitting NN on full set of training data...')
full_fit_nn = fit_nn(train_data_aug, train_labels_aug)
print('Fitting XGB on full set of training data...')
full_fit_xgb = fit_xgb(train_data_aug, train_labels_aug)
print('Fitting KNN on full set of training data...')
full_fit_knn = fit_knn(train_data, train_labels)

# make predictions for the test data using these trained models
print('Making predictions on test data using NN...')
test_pred_nn = pred_nn(full_fit_nn, test_processed)
print('Making predictions on test data using XGB...')
test_pred_xgb = pred_xgb(full_fit_xgb, test_processed)
print('Making predictions on test data using KNN...')
test_pred_knn = pred_knn(full_fit_knn, test_data)

# assemble these predictions into a data frame
test_level_one_data = pd.DataFrame({'nn': test_pred_nn, 'xgb': test_pred_xgb, 'knn': test_pred_knn})


####################################
# make predictions using the trained meta learner
####################################
test_pred_meta = meta_model.predict_proba(test_level_one_data)[:, 1]

# write these predictions to a csv file
out = pd.DataFrame({'Id': test_id, 'Action': test_pred_meta})
out.to_csv('test_pred_meta.csv', index=False)






