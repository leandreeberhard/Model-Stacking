import xgboost as xgb
import numpy as np
from math import floor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# test train split
X_train, X_test, y_train, y_test = train_test_split(train_processed, train_labels, test_size=0.3)
X_train_aug, y_train_aug = augment_training_data(X_train, y_train)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test)


# specify parameters via map
params = {'max_depth': 6, 'eta': 0.8, 'objective': 'binary:logistic'}
num_round = 75
xgb_fitter = xgb.train(params, xgb_train, num_round)
# make prediction
predictions_xgb = xgb_fitter.predict(xgb_test)

# compute metric
print(metrics.roc_auc_score(y_test, predictions_xgb))


# make predictions
xgb_full_train = xgb.DMatrix(train_data_augmented, label=train_labels_augmented)
xgb_full_test = xgb.DMatrix(test_processed.values)

xgb_fitter = xgb.train(params, xgb_full_train, num_round)
predictions_xgb = np.array(xgb_fitter.predict(xgb_full_test))

# write output to file
out = pd.DataFrame({'Id': test_id, 'Action': predictions_xgb.reshape((predictions_xgb.shape[0],))})

out.to_csv('test_pred10.csv', index=False)