import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import deepsurvk
from deepsurvk.datasets import load_rgbsg

X_train, Y_train, E_train = load_rgbsg(partition='training')
X_test, Y_test, E_test = load_rgbsg(partition='testing')

n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]

cols_standardize = ['grade', 'age', 'n_positive_nodes', 'progesterone', 'estrogen']
X_ct = ColumnTransformer([('standardizer', StandardScaler(), cols_standardize)])
X_ct.fit(X_train[cols_standardize])

X_train[cols_standardize] = X_ct.transform(X_train[cols_standardize])
X_test[cols_standardize] = X_ct.transform(X_test[cols_standardize])

Y_scaler = StandardScaler().fit(Y_train)
Y_train['T'] = Y_scaler.transform(Y_train)
Y_test['T'] = Y_scaler.transform(Y_test)

sort_idx = np.argsort(Y_train.to_numpy(), axis=None)[::-1]
X_train = X_train.loc[sort_idx, :]
Y_train = Y_train.loc[sort_idx, :]
E_train = E_train.loc[sort_idx, :]

params = {'n_layers': 1,
          'n_nodes': 8,
          'activation': 'selu',
          'learning_rate': 0.154,
          'decays': 5.667e-3,
          'momentum': 0.887,
          'l2_reg': 6.551,
          'dropout': 0.661,
          'optimizer': 'nadam'}

dsk = deepsurvk.DeepSurvK(n_features=n_features,
                          E=E_train,
                          **params)

loss = deepsurvk.negative_log_likelihood(E_train)
dsk.compile(loss=loss)

callbacks = deepsurvk.common_callbacks()

epochs = 1000
history = dsk.fit(X_train, Y_train,
                  batch_size=n_patients_train,
                  epochs=epochs,
                  callbacks=callbacks,
                  shuffle=False)

deepsurvk.plot_loss(history)

Y_pred_test = np.exp(-dsk.predict(X_test))
c_index_test = deepsurvk.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")

rec_ij = deepsurvk.recommender_function(dsk, X_test, 'horm_treatment')

recommendation_idx, _ = deepsurvk.get_recs_antirecs_index(rec_ij, X_test, 'horm_treatment')

Y_test_original = Y_test.copy(deep=True)
Y_test_original['T'] = Y_scaler.inverse_transform(Y_test)

deepsurvk.plot_km_recs_antirecs(Y_test_original, E_test, recommendation_idx)
