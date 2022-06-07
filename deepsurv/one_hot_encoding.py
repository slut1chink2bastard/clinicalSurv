import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

import Utilities as Utils
import deepsurvk
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"breast.csv"))

df = Utils.filter_col_data(df, ["Age recode with <1 year olds", "Behavior code ICD-O-3",
                                "Breast - Adjusted AJCC 6th T (1988-2015)", "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS tumor size (2004-2015)",
                                "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)",
                                "Histologic Type ICD-O-3", "Laterality", "Breast Subtype (2010+)",
                                "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
                                "Derived HER2 Recode (2010+)", "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)", "Marital status at diagnosis",
                                "End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])
df = pd.get_dummies(df, prefix=["Age recode with <1 year olds", "Behavior code ICD-O-3",
                                "Breast - Adjusted AJCC 6th T (1988-2015)",
                                "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS extension (2004-2015)",
                                "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)",
                                "Histologic Type ICD-O-3", "Laterality", "Breast Subtype (2010+)",
                                "ER Status Recode Breast Cancer (1990+)",
                                "PR Status Recode Breast Cancer (1990+)", "Derived HER2 Recode (2010+)",
                                "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)", "Marital status at diagnosis"],
                    columns=["Age recode with <1 year olds", "Behavior code ICD-O-3",
                             "Breast - Adjusted AJCC 6th T (1988-2015)",
                             "Breast - Adjusted AJCC 6th N (1988-2015)",
                             "Breast - Adjusted AJCC 6th M (1988-2015)", "CS extension (2004-2015)",
                             "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Histologic Type ICD-O-3",
                             "Laterality", "Breast Subtype (2010+)", "ER Status Recode Breast Cancer (1990+)",
                             "PR Status Recode Breast Cancer (1990+)", "Derived HER2 Recode (2010+)",
                             "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                             "Chemotherapy recode (yes, no/unk)", "Marital status at diagnosis"])
for column in df.columns:
    if column:
        print(column)

training_data, testing_data = Utils.train_test_split(df, test_size=0.2)
print("-----------Training Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(training_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(training_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(training_data))
print("-----------The null value summary-----------")
print(training_data.isnull().sum())

print("-----------Testing Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(testing_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(testing_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(testing_data))
print("-----------The null value summary-----------")
print(testing_data.isnull().sum())

X_train = training_data.drop(["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"], axis=1)
print("-----------The X_train row number-----------")
print(Utils.get_data_frame_row_count(X_train))
print("-----------The X_train col number-----------")
print(Utils.get_data_frame_col_count(X_train))

E_train = Utils.filter_col_data(training_data, ["End Calc Vital Status (Adjusted)"])
print("-----------The E_train row number-----------")
print(Utils.get_data_frame_row_count(E_train))
print("-----------The E_train col number-----------")
print(Utils.get_data_frame_col_count(E_train))
E_train = pd.Series(np.where(E_train.iloc[:, 0].values == "Dead", 1, 0), index=E_train.index).T.to_numpy()

Y_train = Utils.filter_col_data(training_data, ["Number of Intervals (Calculated)"])
print("-----------The Y_train row number-----------")
print(Utils.get_data_frame_row_count(Y_train))
print("-----------The Y_train col number-----------")
print(Utils.get_data_frame_col_count(Y_train))

X_test = testing_data.drop(["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"], axis=1)
print("-----------The X_train row number-----------")
print(Utils.get_data_frame_row_count(X_test))
print("-----------The X_train col number-----------")
print(Utils.get_data_frame_col_count(X_test))

E_test = Utils.filter_col_data(testing_data, ["End Calc Vital Status (Adjusted)"])
print("-----------The E_train row number-----------")
print(Utils.get_data_frame_row_count(E_test))
print("-----------The E_train col number-----------")
print(Utils.get_data_frame_col_count(E_test))

E_test = pd.Series(np.where(E_test.iloc[:, 0].values == "Dead", 1, 0), index=E_test.index).T.to_numpy()

Y_test = Utils.filter_col_data(testing_data, ["Number of Intervals (Calculated)"])
print("-----------The Y_train row number-----------")
print(Utils.get_data_frame_row_count(Y_test))
print("-----------The Y_train col number-----------")
print(Utils.get_data_frame_col_count(Y_test))

X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

Y_scaler = StandardScaler().fit(Y_train.to_numpy().reshape(-1, 1))
Y_train = Y_scaler.transform(Y_train)
Y_test = Y_scaler.transform(Y_test)
Y_train = Y_train.flatten()

Y_test = Y_test.flatten()

# %% [markdown]
# > Notice that if you read/have your data as a `pandas` DataFrame, you will
# > get an error when reshaping `Y_train` (see [issue #81](https://github.com/arturomoncadatorres/deepsurvk/issues/81)).
# > That is because a DataFrame doesn't have the `reshape` attribute.
# >
# > In such case, you need to do the reshaping as follows:
# >
# > ```
# > Y_scaler = StandardScaler().fit(Y_train.values.reshape(-1, 1))
# > ```

# %%
# Sorting
sort_idx = np.argsort(Y_train)[::-1]
X_train = X_train[sort_idx]
Y_train = Y_train[sort_idx]
E_train = E_train[sort_idx]

# %% [markdown]
# > Notice that if you read/have your data as a `pandas` DataFrame, you will
# > get an error when sorting (see [issue #82](https://github.com/arturomoncadatorres/deepsurvk/issues/82)).
# > That is because a DataFrame cannot be sorted like this.
# >
# > In such case, you need to do the sorting as follows:
# >
# > ```
# > X_train = X_train.values[sort_idx]
# > ...
# > ```
#
# ## Create a DeepSurvK model
# When creating an instance of a DeepSurvK model, we can also define its
# parameters. The only mandatory parameters are `n_features` and `E`.
# If not defined, the rest of the parameters will use a default.
# This is, of course, far from optimal, since (hyper)parameter tuning
# has a *huge* impact on model performance. However, we will deal
# with that later.

params = {"n_layers": 1,
          "n_nodes": 10,
          "activation": "selu",
          "learning_rate": 0.011,
          "decays": 5.667e-3,
          "momentum": 0.887,
          "l2_reg": 6.551,
          "dropout": 0.661,
          "optimizer": "nadam"}

n_features = X_train.shape[1]
n_patients_train = X_train.shape[0]
# %%
dsk = deepsurvk.DeepSurvK(n_features=n_features, E=E_train, **params)

# %% [markdown]
# Since DeepSurvK is just a Keras model, we can take advantage of all the
# perks and tools that come with it. For example, we can get an overview
# of the model architecture very easily.

# %%
dsk.summary()

# %% [markdown]
# ## Callbacks
# As mentioned earlier, it is practical to use Early Stopping in the
# case of NaNs in loss values. Additionally, it is also a good idea
# to use the model that during the training phase yields the lowest loss
# (which isn't necessarily the one at the end of the training)
#
# Both of these practices can be achieved using [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback).
# DeepSurvK provides a method to generate these two specific callbacks.

# %%
callbacks = deepsurvk.common_callbacks()
print(callbacks)

# %% [markdown]
# Needless to say that you can define your own callbacks as well, of course.
#
# ## Model fitting
# After this, we are ready to actually fit our model (as any Keras model).

# %%
epochs = 10
history = dsk.fit(X_train, Y_train,
                  batch_size=n_patients_train,
                  epochs=epochs,
                  callbacks=callbacks,
                  shuffle=False)

# %% [markdown]
# > In some cases, it has been reported that while fitting a model,
# > the [loss goes to a `NaN` very early](https://github.com/arturomoncadatorres/deepsurvk/issues/83),
# > making the training process unfeasible, even with the previously defined
# > callback. I haven't been able to replicate that issue consistently.
# >
# > However, this issue has also [been reported in the original DeepSurv](https://github.com/jaredleekatzman/DeepSurv/issues/14).
# > Apparently, a potentially good solution for this is to *not*
# > standardize your data during the pre-procesing, but rather
# > normalizing it (i.e., make sure that features are in the range 0-1).
# > However, remember that scaling is particularly sensitive to
# > outliers, so be careful!

# %% [markdown]
# DeepSurvK provides a few wrappers to generate visualizations that are
# often required fast and easy.

# %%
deepsurvk.plot_loss(history)

# %% [markdown]
# ## Model predictions
# Finally, we can generate predictions using our model.
# We can evaluate them using the c-index.

# %%
Y_pred_train = np.exp(-dsk.predict(X_train))
c_index_train = deepsurvk.concordance_index(Y_train, Y_pred_train, E_train)
print(f"c-index of training dataset = {c_index_train}")

Y_pred_test = np.exp(-dsk.predict(X_test))
c_index_test = deepsurvk.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")
