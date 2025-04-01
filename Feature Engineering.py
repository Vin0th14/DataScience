import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Binarizer,Normalizer

data  = pd.read_csv(r'E:/Data science Notes/Python/data sets/sample.csv')
data.head()

# 1) Handling na values in numerical column 
# to know the no of null valeus present
print(data.isnull().sum())
# a) you can fill with meaan if there are less null values
data.fillna(data.mean())
# b) you can drop the rows with more null values
#print(data.isnull().sum())
data.dropna(how='all',subset=['BMI','Age'])  #how=all, then it will drop a row only if all the column are null,
                                             #how=any, then it will drop a row  if any one of the column has null
                                             # subset = col name, it will consider only those columns

# 2) Handling na values in categorical column.
print(data.isnull().mean()) # drop the column if the mean is greater than 0.8
cols_to_dropped=[columns for columns in data.columns if data[columns].isnull().mean()>=0]
print("cols to be dropped",cols_to_dropped)  # you can drop or fill with mode i.e most repeating value
#data1=data.drop(columns=cols_to_dropped,inplace=True)
print(data.info())
print(data['BMI'].describe()) # to get min max std 25% 50%
print(data.mode())

# 3) outlier detection using std
plt.hist(x=data['BMI'],bins=20)
plt.show()
std_dev=data['BMI'].std()
mean=data['BMI'].mean()
print(mean,std_dev)
upperlimit=mean+(3*std_dev)   # values after +-3 std will be considered outliers
lowerlimit=mean-(3*std_dev)

mask=((data['BMI']<lowerlimit)|(data['BMI']>upperlimit))  # to get the outliers
print(data[mask])
r_mask=((data['BMI']>=lowerlimit)&(data['BMI']<=upperlimit)) # to remove the outliers
dataa_rmvdoutlier=data[r_mask]
print(dataa_rmvdoutlier.head(20))
print(data.shape)
print(dataa_rmvdoutlier.shape)

#4) outlier detection using z score

def zscore(df, column):            # Zscore should lie between 3 and -3
    x=df[column].values
    mean=df[column].mean()
    stddev=df[column].std()
    return (x-mean)/stddev

datawithzscore=data
datawithzscore['Zscore_BMI']=zscore(datawithzscore,'BMI')
print(datawithzscore.head())

mask_z=((datawithzscore['Zscore_BMI']<-3)|(datawithzscore['Zscore_BMI']>3))  # to get the outliers
print(datawithzscore[mask_z])
r_mask_z=((datawithzscore['Zscore_BMI']>=-3)&(datawithzscore['Zscore_BMI']<=3)) # to remove the outliers
dataa_rmvdoutlier_z=data[r_mask_z]
print(datawithzscore.shape)
print(dataa_rmvdoutlier_z.shape)

#5) outlier detection using pandas

df=pd.DataFrame({"Values":[5,5,5]+[50]*25+[89,90]})
print(df.head())

plt.hist(df['Values'])
plt.show()

lowerBound=0.1
upperBound=0.95
outlier=df['Values'].quantile([lowerBound,upperBound])
print(outlier)

out_mask=(df['Values']<outlier[lowerBound])|(df['Values']>outlier[upperBound])
print(df[out_mask])   # to get the outliers

df[out_mask] = df[~out_mask].median()["Values"]
print(df)            # to remove the outliers

# 6) Simple imputer

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')    # only numerical columns should be fitted
imputer.fit(data)



