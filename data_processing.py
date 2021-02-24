import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import ceil, floor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')

# check distribution of feature values vs label
'''
for colname in train_data.columns.values:
    if colname != 'ACTION':
        sns.catplot('ACTION', colname, data=train_data)
'''

# add features based on ranges where there were no 0 actions
def add_features(train_data):
    train_data['RR1_range'] = [1 if y >= 150000 else 0 for y in train_data['ROLE_ROLLUP_1']]
    train_data['RR2_range'] = [1 if y >= 140000 else 0 for y in train_data['ROLE_ROLLUP_2']]
    train_data['RC_range'] = [1 if y >= 200000 else 0 for y in train_data['ROLE_CODE']]
    train_data['RF_range'] = [1 if 125000 <= y <= 200000 else 0 for y in train_data['ROLE_FAMILY']]
    train_data['RFD_range'] = [1 if 1000 <= y <= 100000 else 0 for y in train_data['ROLE_FAMILY_DESC']]
    train_data['RT_range'] = [1 if 200000 <= y <= 250000 else 0 for y in train_data['ROLE_TITLE']]
    train_data['RD_range'] = [1 if 50000 <= y <= 100000 else 0 for y in train_data['ROLE_DEPTNAME']]
    train_data['R_range'] = [1 if 150000 <= y <= 250000 else 0 for y in train_data['RESOURCE']]

add_features(train_data)

train_labels = train_data['ACTION']
del train_data['ACTION']

# number of levels in each variable
n_elems = [len(train_data.iloc[:, i].value_counts()) for i in range(9)]
n_elems_per_var = dict(zip(train_data.columns, n_elems))

train_data[['RESOURCE', 'MGR_ID']].groupby('MGR_ID').size()
mgr_id_25_ind = [train_data['MGR_ID'][i] == 25 for i in range(len(train_data['MGR_ID']))]
len(set(train_data[['RESOURCE', 'MGR_ID']][mgr_id_25_ind]['RESOURCE']))

del train_data['ROLE_CODE']


# TEST DATA
test_data = pd.read_csv('test.csv')
test_id = test_data['id']
add_features(test_data)

del test_data['id']
del test_data['ROLE_CODE']

train_data.shape
test_data.shape

# convert data to categorical
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

# combine train and test data into a single data frame for processing
train_test_data = pd.concat([train_data, test_data], ignore_index=True)
train_test_data.shape

########################################
# try to find patterns in the data
########################################

# look at resources with only a single entry
resource_value_count = train_test_data['RESOURCE'].value_counts()
single_entry_elem = set(x for x in resource_value_count.index if resource_value_count[x] == 1)
single_entry_ind = [train_test_data['RESOURCE'][i] in single_entry_elem for i in range(len(train_test_data['RESOURCE']))]
resource_single_entry = train_test_data[single_entry_ind]

resource_single_entry.groupby(by='ROLE_FAMILY').count()

# check that role family is the same among points with the same role family desc
n_roles = train_test_data[['ROLE_FAMILY', 'ROLE_FAMILY_DESC']].groupby('ROLE_FAMILY_DESC').agg({"ROLE_FAMILY": lambda x: x.nunique()})
set([x[0] for x in n_roles.values])


role_fam = pd.DataFrame({'ind': range(len(train_test_data['ROLE_FAMILY'])), 'ROLE_FAMILY': train_test_data['ROLE_FAMILY']})
role_fam.groupby('ROLE_FAMILY').count()
min(role_fam.groupby('ROLE_FAMILY').count().values)

list(role_fam.groupby('ROLE_FAMILY').count().values).index(1)
role_fam.groupby('ROLE_FAMILY').count().iloc[50]

# 'ROLE_FAMILY' 125808 has only one elements
ceo_ind = [train_test_data['ROLE_FAMILY'][i] == 125808 for i in range(train_test_data.shape[0])]
ceo_dict = dict(zip(train_test_data[ceo_ind].columns.tolist(), train_test_data[ceo_ind].values.tolist()[0]))


# example of 'ROLE_FAMILY_DESC' with 9 'ROLE_FAMILY'
ind = list(n_roles.values).index([9])
select_role = n_roles.iloc[ind, :].name

select_ind = [train_test_data['ROLE_FAMILY_DESC'][i] == select_role for i in range(train_test_data.shape[0])]
train_test_data[['ROLE_FAMILY', 'ROLE_FAMILY_DESC']][select_ind]['ROLE_FAMILY'].value_counts()



# example 'ROLE_FAMILY_DESC' = 117886
select_ind = [train_test_data['ROLE_FAMILY_DESC'][i] == 117886 for i in range(train_test_data.shape[0])]
train_test_data[['ROLE_FAMILY', 'ROLE_FAMILY_DESC']][select_ind]['ROLE_FAMILY'].value_counts()


########################################
########################################



train_test_processed = convert_data_to_categorical(train_test_data)

# split back into two separate data frames
train_processed = train_test_processed[:train_data.shape[0]]
test_processed = train_test_processed[train_data.shape[0]:]

assert train_processed.shape[0] == train_data.shape[0] and test_processed.shape[0] == test_data.shape[0]


# augment training data
def augment_training_data(X_train, y_train):
    n_maj, n_min = list(pd.DataFrame(y_train).value_counts())
    # number of times to replicate each minority class member
    n_repeat = floor(n_maj / n_min)

    # minority class indices (indices with 0)
    min_ind = np.equal(y_train, 0)
    X_train_min = X_train[min_ind]
    X_train_aug = np.concatenate([X_train] + [X_train_min]*n_repeat)

    y_train_min = y_train[min_ind]
    y_train_aug = np.concatenate([y_train] + [y_train_min]*n_repeat)

    # shuffle the indices
    perm = np.random.permutation(range(X_train_aug.shape[0]))
    X_train_aug = X_train_aug[perm]
    y_train_aug = y_train_aug[perm]

    return X_train_aug, y_train_aug


train_data_augmented, train_labels_augmented = augment_training_data(train_processed, train_labels)