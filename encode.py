from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def encodeCategoricalColumn(df):
    ohe = OneHotEncoder(drop='first',sparse_output=False)
    ohe_np=ohe.fit_transform(df[['Sex','Cabin','Embarked']])
    ohe_np_array=np.array(ohe_np)
    columns= ohe.get_feature_names_out()

    combined = pd.DataFrame(ohe_np_array,columns=columns)
    return {
        "combined":combined,
        "columns":columns
    }