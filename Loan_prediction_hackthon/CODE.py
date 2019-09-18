#import the library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Imputer

#Load the data set 
df=pd.read_csv("train_ctrUa4K.csv")
df1=pd.read_csv("test_lAUu6dG.csv")

#Data Exploration 
df.head(10)
des=df.describe()
#for non_numerical data
PA=df['Property_Area'].value_counts()
GDN=df['Gender'].value_counts()
MA=df['Married'].value_counts()
EDU=df['Education'].value_counts()
SE=df['Self_Employed'].value_counts()
#by above code we understand what is in the data 

#finding the missing values and null values of numerical data
print (df.isnull().sum())

# Replace using median 
median = df['LoanAmount'].median()
df['LoanAmount'].fillna(median, inplace=True)

median= df['Loan_Amount_Term'].median()
df['Loan_Amount_Term'].fillna(median, inplace=True)

median= df['Credit_History'].median()
df['Credit_History'].fillna(median, inplace=True)

#Remove the null values 
df = df.dropna(how='any',axis=0)

print (df.isnull().sum())

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_df = LabelEncoder()
df["Gender"] = labelencoder_df.fit_transform(df["Gender"])

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_df = LabelEncoder()
df["LoanAmount"] = labelencoder_df.fit_transform(df["LoanAmount"])


labelencoder_df = LabelEncoder()
df["LoanAmount"] = labelencoder_df.fit_transform(df["LoanAmount"])


labelencoder_df = LabelEncoder()
df["Education"] = labelencoder_df.fit_transform(df["Education"])


labelencoder_df = LabelEncoder()
df["Self_Employed"] = labelencoder_df.fit_transform(df["Self_Employed"])



labelencoder_df = LabelEncoder()
df["Property_Area"] = labelencoder_df.fit_transform(df["Property_Area"])


labelencoder_df = LabelEncoder()
df["Loan_Status"] = labelencoder_df.fit_transform(df["Loan_Status"])


labelencoder_df = LabelEncoder()
df["Married"] = labelencoder_df.fit_transform(df["Married"])


labelencoder_df = LabelEncoder()
df["Loan_ID"] = labelencoder_df.fit_transform(df["Loan_ID"])

labelencoder_df = LabelEncoder()
df["Dependents"] = labelencoder_df.fit_transform(df["Dependents"])

#slipt the data
X =df.iloc[:, :-1].values
y =df['Loan_Status'].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Performing the model 
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
#ROUND OF THE pred 
y_pred=y_pred.round(0)

