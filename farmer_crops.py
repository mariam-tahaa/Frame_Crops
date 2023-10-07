#!/usr/bin/env python
# coding: utf-8

# In[13]:


# All required libraries 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


###Reading the data
# 1-Load the dataset
data = pd.read_csv("D:\\machine learning\\python_task\\farmer_regression\\soil_measures.csv")


# In[ ]:


describe= data.describe()
print("Data describtion: \n", describe)


# In[6]:


# 2-check out missing values
missing= data.isnull().sum().max()
print("missing values is: ", missing)


# In[8]:


# 3-check for crops types
crops_types= data['crop'].unique()
print("Crops types: ", crops_types)


# In[9]:


#print data after spliting to X and y
X = data.iloc[: , :-1]
y = data.iloc[: , -1]
print("X data: \n", X.head())
print("Y data: \n", y.head())


# In[10]:


###Splitting the data to training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= .2, random_state= 42, shuffle= True)


# In[11]:


X.shape


# In[14]:


#predict the crop using best features
for features in ["N", "P", "K", "ph"]:
    log_reg= LogisticRegression(max_iter= 2000, multi_class= "multinomial")
    log_reg.fit(X_train[[features]], y_train)
    y_predict= log_reg.predict(X_test[[features]])
    f1= f1_score(y_test, y_predict, average="weighted")
    print(f"F1_Score for {features}: {f1}")


# In[16]:


# 2-Calculate the correlation matrix
cor= data[["N", "P", "K", "ph"]].corr()
sns.heatmap(cor, annot= True)
plt.show()


# In[29]:


final_features= ["P", "K", "N"]


# In[34]:


#split data with new features
X_train, X_test, y_train, y_test= train_test_split(data[final_features], data["crop"], 
                                                   test_size= .2, random_state= 42, shuffle= True)


# In[35]:


#predict types of crops based on new features
log_reg= LogisticRegression(max_iter= 2000, multi_class= "multinomial").fit(X_train, y_train)
y_predict= log_reg.predict(X_test)
model_performance= f1_score(y_test, y_predict, average="weighted")
print("Model Performance= ", model_performance)


# In[ ]:




