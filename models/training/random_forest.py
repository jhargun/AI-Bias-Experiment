#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn


# In[2]:


train_df = pd.read_csv('../dataset/train_preprocessed.csv')
train_df.sample(frac=.3)


# In[3]:


test_df = pd.read_csv('../dataset/test_preprocessed.csv')


# In[4]:


train_df.info()


# Preprocess data

# In[5]:


[col for col in train_df.columns if train_df[col].dtype != bool]


# In[6]:


# Drop unnecessary columns
train_df.drop(['index', 'SERIAL', 'PERNUM', 'HHWT', 'CLUSTER', 'STRATA', 'PERWT','YRMARR', 'YRNATUR', 'carpools'], axis=1, inplace=True)


# In[7]:


[col for col in train_df.columns if train_df[col].dtype == 'object']


# In[8]:


for col in train_df.columns:
    if train_df[col].dtype == 'object':
        unique = train_df[col].unique()
        assert(len(unique) == 3 and True in unique and False in unique)
        print(f"Replacing {train_df[col].isna().values.sum()} NaN values in {col} with false")
        train_df[col] = train_df[col].fillna(False).astype(bool)


# In[9]:


for col in train_df.columns:
    if(train_df[col].isna().values.sum() > 0):
        print("Warning: NaN values in column", col)


# In[10]:


train_df.drop(['TRANTIME'], axis=1, inplace=True)


# Train model

# In[11]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=100, max_depth=7, criterion='absolute_error', random_state=0, n_jobs=-1)


# In[12]:


print("Training")
forest.fit(train_df.drop('INCWAGE_CPIU_2010', axis=1), train_df['INCWAGE_CPIU_2010'])


# In[ ]:




