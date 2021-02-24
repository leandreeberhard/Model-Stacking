import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import ceil, floor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
train_labels = train_data['ACTION']
del train_data['ACTION']
train_data.shape

# role_title and role_code give the same information
# check this
len(set(train_data['ROLE_CODE'].values))
len(set(train_data['ROLE_TITLE'].values))

# check that there is a unique title for each code
for code in set(train_data['ROLE_CODE']):
    code_ind = np.equal(train_data['ROLE_CODE'], code)
    titles = set(train_data[code_ind]['ROLE_TITLE'])
    if len(titles) > 1:
        print(f'Mismatch for code {code}')
        break
    print(f'Code {code} = title {titles}')

del train_data['ROLE_CODE']

train_data.shape

# if there are less than 20 members in a factor level, change the level to a value of zero
for col in range(0, train_data.shape[1]):
    levels = train_data.iloc[:, col].value_counts().index.values

    vals_to_change = [level for level in levels if train_data.iloc[:, col].value_counts()[level] <= 20]

    n_rows = (train_data.iloc[:, col]).shape[0]
    for row in range(n_rows):
        print(f'Column {col}, row {row} of {n_rows}')
        if train_data.iat[row, col] in vals_to_change:
            train_data.iat[row, col] = 0

# encode features as categorical variables
train_categorical = pd.get_dummies(train_data, columns=train_data.columns)
train_categorical.shape

'''
# comress the data with factor analysis
from sklearn.decomposition import FactorAnalysis
compressor = FactorAnalysis(n_components=200)
train_compressed = compressor.fit_transform(train_categorical, y_train)
'''
# check class distribution
train_labels.value_counts()  # data is unbalanced – don't optimize accuracy directly

# test/train split
X_train, X_test, y_train, y_test = train_test_split(train_categorical, train_labels.values, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3)

# duplicate data points from the minority class
# y_train.value_counts()
n_maj, n_min = list(pd.DataFrame(y_train).value_counts())
# number of times to replicate each minority class memeber
n_repeat = floor(n_maj / n_min)

# minority class indices (indices with 0)
min_ind = np.equal(y_train, 0)
X_train_aug = X_train
for _ in range(n_repeat):
    X_train_aug = np.concatenate((X_train_aug, X_train[min_ind]), axis=0)
X_train_aug.shape

y_train_aug = y_train
for _ in range(n_repeat):
    y_train_aug = np.concatenate((y_train_aug, y_train[min_ind]), axis=0)

# shuffle the indices
perm = np.random.permutation(range(X_train_aug.shape[0]))
X_train_aug = X_train_aug[perm, :]
y_train_aug = y_train_aug[perm]

#####################################
# fit a random forest classifier
#####################################
rf = RandomForestClassifier(verbose=1, n_jobs=4, n_estimators=300)

rf.fit(X_train_aug, y_train_aug)

# make predictions using fitted model
predictions = rf.predict(X_test)
# class balance of predictions
pd.DataFrame(predictions).value_counts()

# calculate the cost-adjusted accuracy
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.roc_auc_score(y_test, predictions))

#####################################
# build a neural network to fit
#####################################
import tensorflow as tf

# convert pd dataframes to np arrays
X_train_aug = X_train_aug
y_train_aug = y_train_aug.reshape((y_train_aug.shape[0], 1))

'''
X_train_nn = X_train
y_train_nn = y_train.reshape((y_train.shape[0], 1))
'''

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(X_train_aug.shape[1],), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

'''
# logistic regression
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train_aug.shape[1], ), activation='sigmoid'),
])
'''

model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy', weighted_metrics=['acc'])

history = model.fit(X_train_aug, y_train_aug, batch_size=256, epochs=100, verbose=1, validation_split=0.2,
                    class_weight={1: n_repeat / (n_repeat + 1), 0: 1 / n_repeat})

# make predictions
predictions_nn = [1 if y > 0.5 else 0 for y in model.predict(X_test)]
# check class balance
pd.DataFrame(predictions_nn).value_counts()
# calculate area under ROC curve
print(metrics.classification_report(predictions_nn, y_test))
print(metrics.confusion_matrix(predictions_nn, y_test))
print(metrics.roc_auc_score(predictions_nn, y_test))


############################
# cross validation to get a better estimate of the performance
############################
def convert_train_to_categorical(X_train_raw):
    X_train_processed = X_train_raw.copy(deep=True)

    for col in range(0, X_train_processed.shape[1]):
        levels = X_train_processed.iloc[:, col].value_counts().index.values

        vals_to_change = [level for level in levels if X_train_processed.iloc[:, col].value_counts()[level] <= 20]
        print(f'Column {col}')
        n_rows = (X_train_processed.iloc[:, col]).shape[0]
        for row in range(n_rows):
            if X_train_processed.iat[row, col] in vals_to_change:
                X_train_processed.iat[row, col] = 0

    # encode features as categorical variables
    train_categorical = pd.get_dummies(X_train_processed, columns=X_train_processed.columns)

    return train_categorical, X_train_processed


def convert_test_to_categorical(X_test_raw, X_train_raw):
    X_test_raw = X_test_raw.copy(deep=True)

    for col in range(0, X_test_raw.shape[1]):
        train_levels = set(X_train_raw.iloc[:, col].value_counts().index.values)
        print(f'Column {col}')
        n_rows = (X_test_raw.iloc[:, col]).shape[0]
        for row in range(n_rows):

            if X_test_raw.iat[row, col] not in train_levels:
                X_test_raw.iat[row, col] = 0

    # encode features as categorical variables
    test_categorical = pd.get_dummies(X_test_raw, columns=X_test_raw.columns)

    return test_categorical


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


def fit_nn(X_train, y_train, X_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=(X_train_aug.shape[1],), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', weighted_metrics=['acc'])

    history = model.fit(X_train, y_train, batch_size=256, epochs=10, verbose=1,
                        class_weight={1: n_repeat / (n_repeat + 1), 0: 1 / n_repeat})

    # make predictions
    predictions_nn = [1 if y > 0.5 else 0 for y in model.predict(X_test)]

    if len(set(y_test)) == 1:
        y_test.iloc[0, 0] = abs(1 - y_test.iloc[0, 0])

    return metrics.roc_auc_score(y_test, predictions_nn)


# generate the folds
n_folds = 10
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=n_folds, shuffle=True)

sum = 0
counter = 0
for train_ind, test_ind in k_fold.split(train_data):
    X_train_raw = pd.DataFrame(train_data.values[test_ind])
    y_train = pd.DataFrame(train_labels.values[test_ind])
    X_test_raw = pd.DataFrame(train_data.values[train_ind])
    y_test = pd.DataFrame(train_labels.values[train_ind])

    # carry out the feature processing separately on training and test data
    X_train, X_train_processed = convert_train_to_categorical(X_train_raw)
    X_test = convert_test_to_categorical(X_test_raw, X_train_processed)

    print('Augmenting training data...')
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train)

    sum += fit_nn(X_train_aug, y_train_aug, X_test, y_test)
    counter += 1
    print(f'Loop: {counter} of {n_folds}, Estimated Performance Metric: {sum / counter}')

# estimate of the metric
print(sum / n_folds)

###################################
# kernelized svm
###################################
from sklearn.svm import SVC

y_train_svm = y_train_aug.reshape((y_train_aug.shape[0],))
X_train_svm = X_train_aug

svm_fitter = SVC(kernel='rbf', class_weight={1: n_repeat / (n_repeat + 1), 0: 1 / n_repeat}, probability=True)
svm_fitter.fit(X_train_svm, y_train_svm)

# predictions for test data
predictions_svm = svm_fitter.predict(X_test)
# calculate area under ROC curve
print(metrics.classification_report(y_test, predictions_svm))
print(metrics.confusion_matrix(y_test, predictions_svm))
print(metrics.roc_auc_score(y_test, predictions_svm))


############################
# process test data and make predictions using fitted model
############################
test_data = pd.read_csv('test.csv')
test_id = test_data['id']
del test_data['id']
del test_data['ROLE_CODE']

train_data.shape

# if a level if not present in the test set, reassign the level to 0
for col in range(0, test_data.shape[1]):
    train_levels = set(train_data.iloc[:, col].value_counts().index.values)

    n_rows = (test_data.iloc[:, col]).shape[0]
    for row in range(n_rows):
        print(f'Column {col}, row {row} of {n_rows}')
        if test_data.iat[row, col] not in train_levels:
            test_data.iat[row, col] = 0

# encode features as categorical variables
test_categorical = pd.get_dummies(test_data, columns=test_data.columns)
test_categorical.shape

# make predictions
predictions_svm_test = svm_fitter.predict(test_categorical)
# predictions_svm_test = model.predict(test_categorical)
# predictions_svm_test = [1 if y > 0.5 else 0 for y in predictions_svm_test]


# check output class balance
pd.DataFrame(predictions_svm_test).value_counts()

# write results to a csv
out = pd.DataFrame({'Id': test_id, 'Action': predictions_svm_test})

out.to_csv('test_pred2.csv', index=False)
