from matplotlib import pyplot as plt
import seaborn as sns

def checkAndRemoveOutliers(df):
        checkOutliers(df)
        cleaned_data = df.copy()
        for col in df.columns:
           free_outlier=detectAndRemoveOutliers(df[col])
           if free_outlier.any():
                cleaned_data=cleaned_data[free_outlier]
                
           return cleaned_data
        checkOutliers(cleaned_data)



def detectAndRemoveOutliers(column):
    q1=column.quantile(0.25)
    q3=column.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    higher = q3 + 1.5 * iqr

    new_data = (column > lower) & (column < higher)
    return new_data   



def checkOutliers(df):
    for col in df.columns:
        sns.boxplot(df[col])
        plt.show()
