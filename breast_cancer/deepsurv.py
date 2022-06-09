# # Cox-PH and DeepSurv
#
# In this script we will train the [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1).
# I will use my own dataset from SEER
#
# A more detailed introduction to the `pycox` package can be found in [this notebook](https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb) about the `LogisticHazard` method.
#
# The main benefit Cox-CC (and the other Cox methods) has over Logistic-Hazard is that it is a continuous-time method, meaning we do not need to discretize the time scale.

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import utilities as Utils

import torch
import torchtuples as tt

import breast_utilities as br_utils
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(999)

# ## Dataset
#
# We load the METABRIC data set and split in train, test and validation.

# In[4]:

df = Utils.read_from_file("data/breast.csv")
df = Utils.filter_col_data(df, ["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                "ICD-O-3 Hist/behav",
                                "Breast - Adjusted AJCC 6th T (1988-2015)", "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                "Laterality", "Breast Subtype (2010+)",
                                "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)",
                                "End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])

# take a look of the data info
Utils.print_data_frame_info(df)

# according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
duct_carcinoma_array = ['8500/3: Infiltrating duct carcinoma, NOS', '8501/3: Comedocarcinoma, NOS',
                        '8502/3: Secretory carcinoma of breast',
                        '8503/3: Intraductal papillary adenocarcinoma with invasion',
                        '8504/3: Intracystic carcinoma, NOS', '8507/3: Ductal carcinoma, micropapillary']
# according to https://seer.cancer.gov/icd-o-3/sitetype.icdo3.20220429.pdf
lobular_and_other_ductal_array = ['8520/3: Lobular carcinoma, NOS', '8521/3: Infiltrating ductular carcinoma',
                                  '8522/3: Infiltrating duct and lobular carcinoma',
                                  '8523/3: Infiltrating duct mixed with other types of carcinoma',
                                  '8524/3: Infiltrating lobular mixed with other types of carcinoma',
                                  '8525/3: Polymorphous low grade adenocarcinoma']
duct_lobular_array = duct_carcinoma_array + lobular_and_other_ductal_array

# filter the ICD-O-3 Hist/behav whose type is DUCT CARCINOM and LOBULAR AND OTHER DUCTAL CA
df = Utils.select_data_from_values(df, "ICD-O-3 Hist/behav", duct_lobular_array)

# map "RX Summ--Surg Prim Site (1998+)" according to map_breast_surg_type
df = Utils.map_one_col_data(df, "RX Summ--Surg Prim Site (1998+)", br_utils.map_breast_surg_type)

# map "End Calc Vital Status (Adjusted)" according to map_event_code
df = Utils.map_one_col_data(df, "End Calc Vital Status (Adjusted)", br_utils.map_event_code)

# take a look of the data info again
print("------------------After filtering and Mapping------------------")
Utils.print_data_frame_info(df)
df = pd.get_dummies(df, prefix=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                "ICD-O-3 Hist/behav",
                                "Breast - Adjusted AJCC 6th T (1988-2015)",
                                "Breast - Adjusted AJCC 6th N (1988-2015)",
                                "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                "Laterality", "Breast Subtype (2010+)",
                                "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                "Chemotherapy recode (yes, no/unk)"],
                    columns=["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                             "ICD-O-3 Hist/behav",
                             "Breast - Adjusted AJCC 6th T (1988-2015)",
                             "Breast - Adjusted AJCC 6th N (1988-2015)",
                             "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                             "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                             "Laterality", "Breast Subtype (2010+)",
                             "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                             "Chemotherapy recode (yes, no/unk)"])
full_data = df

test_data = Utils.split_data(full_data, 0.2)
Utils.remove_data(full_data, test_data.index)
print("-----------Testing Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(test_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(test_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(test_data))
print("-----------The null value summary-----------")
print(test_data.isnull().sum())

validate_data = Utils.split_data(full_data, 0.1)
Utils.remove_data(full_data, validate_data.index)
print("-----------Validating Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(validate_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(validate_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(validate_data))
print("-----------The null value summary-----------")
print(validate_data.isnull().sum())

train_data = full_data
print("-----------Training Data-----------")
print("-----------The row number-----------")
print(Utils.get_data_frame_row_count(validate_data))
print("-----------The col number-----------")
print(Utils.get_data_frame_col_count(validate_data))
print("-----------The column names are-----------")
print(Utils.get_data_frame_col_names(validate_data))
print("-----------The null value summary-----------")
print(validate_data.isnull().sum())

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# In[5]:


df_train.head()

# ## Feature transforms
# We have 9 covariates, in addition to the durations and event indicators.
#
# We will standardize the 5 numerical covariates, and leave the binary variables as is. As variables needs to be of type `'float32'`, as this is required by pytorch.

# In[6]:


cols_standardize = []
cols_leave = Utils.get_data_frame_col_names_list(train_data)

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

# In[7]:


x_train = x_mapper.fit_transform(train_data).astype('float32')
x_val = x_mapper.transform(validate_data).astype('float32')
x_test = x_mapper.transform(test_data).astype('float32')

# We need no label transforms

# In[8]:


y_train = Utils.get_col_values(train_data,"Number of Intervals (Calculated)"),Utils.get_col_values(train_data,"End Calc Vital Status (Adjusted)")
y_val = Utils.get_col_values(validate_data,"Number of Intervals (Calculated)"),Utils.get_col_values(validate_data,"End Calc Vital Status (Adjusted)")
durations_test = Utils.get_col_values(test_data,"End Calc Vital Status (Adjusted)")
events_test = Utils.get_col_values(test_data,"End Calc Vital Status (Adjusted)")
val = x_val, y_val

# ## Neural net
#
# We create a simple MLP with two hidden layers, ReLU activations, batch norm and dropout.
# Here, we just use the `torchtuples.practical.MLPVanilla` net to do this.
#
# Note that we set `out_features` to 1, and that we have not `output_bias`.

# In[9]:


in_features = x_train.shape[1]
num_nodes = [62, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

# ## Training the model
#
# To train the model we need to define an optimizer. You can choose any `torch.optim` optimizer, but here we instead use one from `tt.optim` as it has some added functionality.
# We use the `Adam` optimizer, but instead of choosing a learning rate, we will use the scheme proposed by [Smith 2017](https://arxiv.org/pdf/1506.01186.pdf) to find a suitable learning rate with `model.lr_finder`. See [this post](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6) for an explanation.

# In[10]:


model = CoxPH(net, tt.optim.Adam)

# In[11]:


batch_size = 10000
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()

# In[12]:


lrfinder.get_best_lr()

# Often, this learning rate is a little high, so we instead set it manually to 0.01

# In[13]:


model.optimizer.set_lr(0.01)

# We include the `EarlyStopping` callback to stop training when the validation loss stops improving. After training, this callback will also load the best performing model in terms of validation loss.

# In[14]:


epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

# In[15]:


log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)

# In[16]:


_ = log.plot()

# We can get the partial log-likelihood

# In[17]:


model.partial_log_likelihood(*val).mean()

# ## Prediction
#
# For evaluation we first need to obtain survival estimates for the test set.
# This can be done with `model.predict_surv` which returns an array of survival estimates, or with `model.predict_surv_df` which returns the survival estimates as a dataframe.
#
# However, as `CoxPH` is semi-parametric, we first need to get the non-parametric baseline hazard estimates with `compute_baseline_hazards`.
#
# Note that for large datasets the `sample` argument can be used to estimate the baseline hazard on a subset.

# In[18]:


_ = model.compute_baseline_hazards()

# In[19]:
model.predict(x_test)

surv = model.predict_surv_df(x_test)

# In[20]:


surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

# ## Evaluation
#
# We can use the `EvalSurv` class for evaluation the concordance, brier score and binomial log-likelihood. Setting `censor_surv='km'` means that we estimate the censoring distribution by Kaplan-Meier on the test set.

# In[21]:


ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

# In[22]:


td = ev.concordance_td()
print(td)

# In[23]:


time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()

# In[24]:


ev.integrated_brier_score(time_grid)

# In[25]:


ev.integrated_nbll(time_grid)

# In[ ]:
