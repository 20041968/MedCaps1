#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from gensim.models import Word2Vec


# In[79]:


get_ipython().system('pip install textblob')


# In[80]:


get_ipython().system('pip install ciso8601')


# In[81]:


df_mental1 = pd.read_csv('mentalhealth.csv', sep='|')


# In[82]:


df_mental1.head()


# In[83]:


df_mental1.info()


# In[84]:


df_mental2 =pd.read_csv('mentalhealth (1).csv', sep='|')


# In[85]:


df_mental2.head()


# In[86]:


df_mental2.info


# In[87]:


df_suicidew1 =pd.read_csv('SuicideWatch.csv',sep='|')


# In[88]:


df_suicidew1.head()


# In[89]:


df_suicidew2 =pd.read_csv('SuicideWatch (1).csv',sep='|')


# In[90]:


df_suicidew2.head()


# In[91]:


sw = pd.DataFrame(df_suicidew1)
mh = pd.DataFrame(df_mental1)
sw['type'] = 'sw'
mh['type'] = 'mh'
df = pd.concat([sw,mh])


# In[92]:


import ciso8601
import calendar

unix_time = calendar.timegm(ciso8601.parse_datetime("1012019").timetuple())


# In[93]:


df_mental1['author'].unique()


# In[94]:


df_suicidew1['author'].unique()


# In[95]:


df_mental2['author'].unique()


# In[96]:


df_suicidew2['author'].unique()


# In[97]:


df.any


# In[98]:


from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")


# In[99]:


print(sentiment_analysis("Im tired"))


# In[100]:


print(sentiment_analysis("I feel alone"))


# In[ ]:




