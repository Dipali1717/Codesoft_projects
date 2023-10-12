#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[2]:


flower_data = pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (5)\IRIS.csv")
flower_data.head()


# In[3]:


flower_data.shape


# In[4]:


flower_data.describe()


# In[5]:


flower_data.info()


# In[6]:


train,test = train_test_split(flower_data,test_size=0.2)


# In[7]:


train


# In[8]:


test


# In[9]:


train_x = train[['sepal_length','sepal_width','petal_length','petal_width']]
train_y = train.species


# In[10]:


train_x


# In[11]:


train_y


# In[12]:


test_x = test[['sepal_length','sepal_width','petal_length','petal_width']]
test_y = test.species


# In[13]:


test_x


# In[14]:


test_y


# In[15]:


model = LogisticRegression()


# In[16]:


model.fit(train_x,train_y)


# In[17]:


pred = model.predict(test_x)
pred


# In[18]:


metrics.accuracy_score(pred,test_y)


# In[22]:


model.DecisionTreeClassifier()


# In[23]:


pred1 = model.predict(train_x)
pred1


# In[24]:


metrics.accuracy_score(pred1,train_y)


# In[ ]:




