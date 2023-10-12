#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


sales_data = pd.read_csv(r"C:\Users\LENOVO\Downloads\advertising.csv")
sales_data.head()


# In[3]:


sales_data.shape


# In[4]:


sales_data.info()


# In[5]:


sales_data.describe()


# In[6]:


sales_data.columns.values


# In[7]:


sales_data.isnull().sum()


# In[8]:


sns.boxplot(data=sales_data, palette='rainbow', orient='h')
plt.tight_layout()


# In[9]:


sns.pairplot(sales_data,x_vars = ['TV', 'Radio', 'Newspaper'],y_vars='Sales',kind='scatter')


# In[10]:


sales_data['TV'].plot.hist(bins=10,alpha=0.5)


# In[11]:


sales_data['Radio'].plot.hist(bins=10,alpha=0.5,color='green')


# In[12]:


sales_data['Newspaper'].plot.hist(bins=10,alpha=0.5,color='red')


# In[13]:


sns.heatmap(sales_data.corr(),annot=True)


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sales_data[['TV']],sales_data[['Sales']],test_size=0.3,random_state=0)


# In[15]:


x_train.head()


# In[16]:


y_train.head()


# In[17]:


x_test.head()


# In[18]:


y_test.head()


# In[19]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[20]:


res = model.predict(x_train)
res


# In[21]:


model.coef_


# In[22]:


model.intercept_


# In[23]:


0.05473199*69.2+7.14382225


# In[24]:


plt.plot(res)


# In[25]:


plt.scatter(x_test,y_test)
plt.plot(x_test,7.14382225+0.05473199*x_test,'r')


# In[ ]:





# In[ ]:




