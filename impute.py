from sklearn.impute import SimpleImputer
import numpy as np

def imputeMissingValues(df):
    si_mean=SimpleImputer(strategy='mean')
    si_mode=SimpleImputer(strategy='most_frequent')
    si_mean_age=si_mean.fit_transform(df[['Age']])
    si_cabin_embarked=si_mode.fit_transform(df[['Cabin','Embarked']])  

    return np.concatenate([df['Sex'].values.reshape(-1,1),si_mean_age,df['Fare'].values.reshape(-1,1),si_cabin_embarked],axis=1)
 