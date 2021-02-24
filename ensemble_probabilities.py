# take the average of predictions contained in the ensemble folder
import pandas as pd
import os

# load data from the files
os.chdir('ensemble')

preds_dict = {}
index = 0
for file in os.listdir():
    if file != '.DS_Store':
        preds_dict[index] = pd.read_csv(file)
        index += 1

n_files = len(preds_dict.keys())

id = preds_dict[0].iloc[:, 0]

preds_cols = [preds_dict[i].iloc[:, 1].values for i in range(n_files)]

preds_average = sum(preds_cols) / n_files

# write the output
out = pd.DataFrame({'Id': id, 'Action': preds_average.reshape((preds_average.shape[0],))})

os.chdir('..')
out.to_csv('test_pred_ens8.csv', index=False)