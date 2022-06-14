import tensorflow as tf
import utilities as Utils
import pandas as pd
import numpy as np
import breast_utilities as br_utils
import matplotlib.pyplot as plt

df = Utils.read_from_file("data/breast.csv")
df = Utils.filter_col_data(df[:50], ["Age recode with <1 year olds", "Marital status at diagnosis", "Grade (thru 2017)",
                                     "ICD-O-3 Hist/behav",
                                     "Breast - Adjusted AJCC 6th T (1988-2015)",
                                     "Breast - Adjusted AJCC 6th N (1988-2015)",
                                     "Breast - Adjusted AJCC 6th M (1988-2015)", "CS Tumor Size/Ext Eval (2004-2015)",
                                     "CS Reg Node Eval (2004-2015)", "CS Mets Eval (2004-2015)",
                                     "Laterality", "Breast Subtype (2010+)",
                                     "RX Summ--Surg Prim Site (1998+)", "Radiation recode",
                                     "Chemotherapy recode (yes, no/unk)",
                                     "End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])
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
Y_train = Utils.filter_col_data(df, ["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"])
X_train = df.drop(["End Calc Vital Status (Adjusted)", "Number of Intervals (Calculated)"], axis=1)
Y_train = Utils.map_one_col_data(Y_train, "End Calc Vital Status (Adjusted)", br_utils.map_event_code)
'''
The S(t) is derived from h(t|X), 
based on https://www.andrew.cmu.edu/user/georgech/Introduction%20to%20Survival%20Analysis%20Math.pdf
Just some integration and sigma opeartions
reference https://github.com/havakv/pycox
'''
model = tf.keras.models.load_model('test.h5', compile=False)
max_duration = np.inf
base_haz = Y_train.assign(expg=np.exp(model.predict(X_train))).groupby("Number of Intervals (Calculated)").agg(
    {'expg': 'sum', "End Calc Vital Status (Adjusted)": 'sum'}).sort_index(ascending=False).assign(
    expg=lambda x: x['expg'].cumsum()).pipe(lambda x: x["End Calc Vital Status (Adjusted)"] / x['expg']).fillna(
    0.).iloc[::-1].loc[lambda x: x.index <= max_duration].rename('baseline_hazards')
base_cum_haz = (base_haz
                .cumsum()
                .rename('baseline_cumulative_hazards'))
base_cum_haz = base_cum_haz.loc[lambda x: x.index <= max_duration]
X_test = X_train[:1]
expg = np.exp(model.predict(X_test)).reshape(1, -1)
base_cum_haz_test = pd.DataFrame(base_cum_haz.values.reshape(-1, 1).dot(expg), index=base_cum_haz.index)
surv = np.exp(-base_cum_haz_test)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()
