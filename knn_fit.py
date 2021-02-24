########################################
# KNN classifier
########################################
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3)

knn_fitter = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='hamming', n_jobs=-1)
knn_fitter.fit(X_train, y_train)
predictions_knn = knn_fitter.predict_proba(X_test)
predictions_knn_one_class = predictions_knn[:, 1]

# performance metrics
print(metrics.roc_auc_score(y_test, predictions_knn_one_class))


# grid search for best parameters
class modifiedKNeighborsClassifier(KNeighborsClassifier):
    def predict(self, X):
        return self.predict_proba(X)[:, 1]


from sklearn.model_selection import GridSearchCV

# inner CV to select parameter
param_grid = {'n_neighbors': [5, 10, 20, 30, 40, 50]}
grid_searcher = GridSearchCV(estimator=modifiedKNeighborsClassifier(metric='hamming', weights='distance'),
                             scoring=metrics.make_scorer(metrics.roc_auc_score),
                             param_grid=param_grid, n_jobs=-1, verbose=1, cv=10)
grid_searcher.fit(train_data, train_labels)
grid_searcher.best_params_['n_neighbors']

# outer CV to estimate performance
from sklearn.model_selection import cross_validate

perf_cv = cross_validate(X=train_data, y=train_labels,
                         estimator=modifiedKNeighborsClassifier(metric='hamming',
                                                                weights='distance',
                                                                n_neighbors=grid_searcher.best_params_['n_neighbors']),
                         scoring=metrics.make_scorer(metrics.roc_auc_score),
                         n_jobs=-1, verbose=1, cv=50)

sum(perf_cv['test_score']) / len(perf_cv['test_score'])

# fit the test data and make predictions
knn_fitter = modifiedKNeighborsClassifier(n_neighbors=10, weights='distance', metric='hamming', n_jobs=-1)
knn_fitter.fit(train_data, train_labels)
predictions_knn = knn_fitter.predict(test_data)
# predictions_knn_one_class = predictions_knn[:, 1]

out = pd.DataFrame({'Id': test_id, 'Action': predictions_knn.reshape((predictions_knn.shape[0],))})

out.to_csv('test_pred9.csv', index=False)
########################################
########################################