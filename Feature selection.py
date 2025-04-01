import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore

df=pd.read_csv(r"D:\Data  Science Interview Prep - Ascendeum\Round 2 ML\LR - Titanic\train.csv")
df.head()

df.shape
print(df.info())
df.duplicated().sum()

# 1 
print(df['Survived'].value_counts())

print("No of null values :", df.isnull().sum() )