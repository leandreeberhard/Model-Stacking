import tensorflow as tf
from math import floor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# test train split
X_train, X_test, y_train, y_test = train_test_split(train_processed, train_labels, test_size=0.3)
X_train_aug, y_train_aug = augment_training_data(X_train, y_train)


n_maj, n_min = list(pd.DataFrame(y_train).value_counts())
# number of times to replicate each minority class memeber
n_repeat = floor(n_maj / n_min)

def get_nn_predictions(X_train_aug, y_train_aug, X_test, n_repeat):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=(X_train_aug.shape[1], ), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss='binary_crossentropy', weighted_metrics=['acc'])

    model.fit(X_train_aug, y_train_aug, batch_size=256, epochs=10, verbose=1, validation_split=0.2,
              class_weight={1: (n_repeat+1) / (n_repeat+2), 0: 1/(n_repeat+2)})

    # make predictions
    predictions_nn = [1 if y > 0.5 else 0 for y in model.predict(X_test)]
    # maybe better to keep the predictions as probabilities
    predictions_nn = model.predict(X_test)
    return predictions_nn

predictions_nn = get_nn_predictions(X_train_aug, y_train_aug, X_test, n_repeat)


# compute AUC score
print(metrics.roc_auc_score(y_test, predictions_nn))
# 0.6246162237412713 with augmented data
# 0.5648230202127912 without augmented data

print(metrics.classification_report(y_test, predictions_nn))
print(metrics.confusion_matrix(y_test, predictions_nn))









# check output class balance
pd.DataFrame(predictions_nn).value_counts()

# fit on full training set
train_data_aug, train_labels_aug = augment_training_data(train_processed, train_labels)
predictions_nn = get_nn_predictions(train_data_aug, train_labels_aug, test_processed, n_repeat)


# write results to a csv
out = pd.DataFrame({'Id': test_id, 'Action': predictions_nn.reshape((predictions_nn.shape[0], ))})

out.to_csv('test_pred12.csv', index=False)









