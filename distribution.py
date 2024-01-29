from matplotlib import pyplot as plt
import seaborn as sns

def checkDistribution(df):
    for col in df.columns:
        plt.figure()
        sns.distplot(df[col])
        plt.show()
