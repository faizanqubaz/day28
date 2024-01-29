import pandas as pd
import numpy as np
from distribution import checkDistribution
from impute import imputeMissingValues
from sklearn.model_selection import train_test_split
from outliers import checkAndRemoveOutliers
from encode import encodeCategoricalColumn
from scaling import scaling
# READ THE DATASET
data = pd.read_csv(r"C:/Users/CL/Desktop/titanic.csv")
data.drop(columns=['PassengerId','Pclass','Name','SibSp','Parch','Ticket'],inplace=True)

X_train,X_test,Y_train,Y_test=train_test_split(data.iloc[:,1:6],data.iloc[:,0],test_size=0.2,random_state=42)

numerical_columns = [col for col in X_train if X_train[col].dtypes == 'float64']
cat_columns = [col for col in X_train if X_train[col].dtypes == 'object']


checkDistribution(X_train[numerical_columns])

# IMPUTE THE MISSING VALUES
missing_new_data_np=imputeMissingValues(X_train)
missing_new_data_farme=pd.DataFrame(missing_new_data_np,columns=X_train.columns)
# print(missing_new_data_farme)
# OUTLIERS DETECTION

new_data=checkAndRemoveOutliers(missing_new_data_farme[numerical_columns])

combined = pd.concat([missing_new_data_farme['Sex'],new_data['Age'],new_data['Fare'],missing_new_data_farme['Cabin'],missing_new_data_farme['Embarked']],axis=1)



# ENCODING THE CATEGORICALCOLUMNS

encode=encodeCategoricalColumn(combined[cat_columns])
others = combined.drop(columns=['Sex','Cabin','Embarked'])


encoding_data=pd.concat([encode['combined'],others],axis=1)

# FEATURE SCALING THE ALL COLUMNS

scalar_data=scaling(encoding_data)


scalar_datafram = pd.DataFrame(scalar_data,columns =['Age','Fare'] + list(encode['columns']))
print(scalar_datafram)