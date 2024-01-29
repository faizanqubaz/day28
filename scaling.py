from sklearn.preprocessing import StandardScaler

def scaling(df):
    
    sc=StandardScaler()
    data=sc.fit_transform(df)
    return data