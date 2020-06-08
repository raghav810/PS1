import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.preprocessing import StandardScaler  #for feature scaling
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing

dataset = pd.read_csv('WGC_10000.csv') # to import the dataset into a variable

#dataset.replace('', np.nan, inplace=True)#set all empty cells as np.nan

dataset= dataset.dropna(axis=1, thresh= 1)
dataset = dataset.dropna(axis=0, thresh= 4)
#dataset_droppedrows = dataset

#only_na = dataset_droppedrows[~dataset_droppedrows.index.isin(dataset_droppedrows.index)]

#dataset_ip = dataset.fillna(dataset.interpolate())
#dataset_ip.to_csv("datasett_ip.csv") #change to mean/median/interpolate based on what you want
dataset = dataset.fillna(dataset.interpolate())

#dataset_ip = dataset_ip.drop(dataset_ip.std()[dataset_ip.std() == 0].index.values, axis=1)#drop columns with 0 variance
#dataset_ip.to_csv("datasett_ip.csv")

dataset = dataset.loc[:, (dataset!= 0).any(axis=0)]#drop the two 0 valued columns

dataset = dataset[dataset.astype('bool').mean(axis=1)>=0.25]#deletes all rows with 75% zeroes
dataset_ip = dataset

dataset.iloc[:,1:-1] = dataset.iloc[:,1:-1].apply(lambda x: ((x-x.min())/(x.max()-x.min())), axis=0) #normalise data between 0 and 1
Corr = dataset.corr()


