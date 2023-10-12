#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[94]:


titanic_data = pd.read_csv(r'C:\Users\LENOVO\Downloads\archive\tested.csv')
titanic_data.head()


# In[95]:


titanic_data.shape


# In[96]:


titanic_data.info()


# In[97]:


titanic_data.isnull().sum()


# In[ ]:





# In[ ]:





# In[98]:


titanic_data.isnull().sum()


# In[99]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
titanic_data['Age']


# In[100]:


print(titanic_data['Fare'].mode())


# In[101]:


titanic_data['Fare'].fillna(titanic_data['Fare'].mode(),inplace=True)


# In[ ]:





# In[102]:


titanic_data.isnull().sum()


# In[103]:


titanic_data.isnull().sum()


# In[104]:


titanic_data.describe()


# In[105]:


titanic_data['Survived'].value_counts()


# In[106]:


sns.set()


# In[107]:


sns.countplot('Survived',data=titanic_data)


# In[108]:


titanic_data['Sex'].value_counts()


# In[109]:


sns.countplot('Sex',data=titanic_data)


# In[110]:


sns.countplot('Sex',hue='Survived',data=titanic_data)


# In[111]:


sns.countplot('Pclass',data=titanic_data)


# In[112]:


sns.countplot('Pclass',hue='Survived',data=titanic_data)


# In[113]:


titanic_data['Sex'].value_counts()


# In[114]:


titanic_data['Embarked'].value_counts()


# In[115]:


titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[116]:


titanic_data.head()


# In[117]:


x = titanic_data[['Pclass','Sex']]
y = titanic_data['Survived']


# In[118]:


x.head()


# In[119]:


y.head()


# In[120]:


print(x.shape,x_train.shape,x_test.shape)


# In[121]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[122]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[123]:


model.fit(x_train,y_train)


# In[124]:


print(model.predict(x_test))


# In[125]:


print(y_test)


# In[129]:


import warnings
warnings.filterwarnings('ignore')
res = model.predict([[3,4]])
if(res==0):
    print("Not survived")
else:
    print("survived")


# In[ ]:





# In[ ]:




