#!/usr/bin/env python
# coding: utf-8

# ## Market Basket Analysis Project

# In[60]:


pip install mlxtend


# In[61]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
import squarify
import networkx as nx


# In[62]:


import warnings
warnings.filterwarnings('ignore')


# In[63]:


groceries=pd.read_csv("C:/Users/USER/OneDrive/Desktop/MTH552A Project/Groceries dataset/Groceries data.csv")
groceries


# In[64]:


groceries.drop(['Date'], axis=1, inplace=True)
groceries


# In[65]:


groceries.groupby('itemDescription').size().sort_values(ascending=False)


# In[82]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[67]:


sns.countplot(groceries['day_of_week'] )


# In[68]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==0).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Sunday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[69]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==1).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Monday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[70]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==2).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Tuesday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[71]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==3).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Wednesday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[72]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==4).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Thursday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[73]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==5).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Friday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[74]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
groceries['itemDescription'].where(groceries['day_of_week']==6).value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items on Saturday', fontsize = 20)
plt.xticks(rotation = 90 )
#plt.grid()
plt.show()


# In[75]:


groceries['date'] = pd.to_datetime(groceries[['year', 'month', 'day']])
groceries['member_date'] = list(zip(groceries['Member_number'], groceries['date'].dt.date))
groceries['quantity'] = 1
groceries


# In[76]:


basket = groceries.groupby(['member_date', 'itemDescription'])['quantity'].count().unstack().fillna(0)
basket


# In[77]:


def convert_values(value):
    if value >= 1:
        return 1
    else:
        return 0

basket = basket.applymap(convert_values)
basket


# In[78]:


basket_items = apriori(basket, min_support = 0.005, use_colnames = True, max_len = 2)
basket_items


# In[79]:


basket_items['length'] = basket_items['itemsets'].apply(lambda x: len(x))
basket_items


# In[80]:


rules = association_rules(basket_items, metric = "confidence", min_threshold=0.13)
rules.sort_values("confidence", ascending=False)


# In[ ]:




