#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


#Load the excel file which has extension of .xlsx
data = pd.read_excel("D:\Data Science Projects\ecom customer_data.xlsx")
print(data)


# In[5]:


data.head()


# In[6]:


df=data.copy()
df.info()


# In[7]:


df.describe()


# # Data Cleaning

# In[8]:


#Check the duplicates
df[df.duplicated()]


# In[9]:


df.isna().sum()


# In[10]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])


# In[11]:


df.isna().sum().sum()


# # Data Visualization
# 

# In[12]:


df.Gender.value_counts()


# In[13]:


sns.countplot(data=df,x='Gender')
plt.show()


# In[14]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(data=df,x='Orders')


# In[15]:


#Order count by each number
plt.subplot(1,2,2)
sns.countplot(data=df,x='Orders',hue='Gender')
plt.suptitle("Overall Orders VS Gender wise Orders")
plt.show()


# In[16]:


#Orders and searches of each brands
cols = list(df.columns[2:])
def dist_list(lst):
    plt.figure(figsize=(30,30))
    for i, col in enumerate(lst,1):
        plt.subplot(6,6,i)
        sns.boxplot(data=df,x=df[col])
dist_list(cols)


# In[17]:


#Heatmap
plt.figure(figsize=(20,15))
sns.heatmap(df.iloc[:,3:].corr())
plt.show()


# In[18]:


df.iloc[:2,:].hist(figsize=(40,30))
plt.show()


# In[19]:


new_df=df.copy()
new_df['Total Search']=new_df.iloc[:,3:].sum(axis=1)


# In[20]:


new_df.sort_values('Total Search', ascending=False)


# In[21]:


plt.figure(figsize=(13,8))
plt_data=new_df.sort_values('Total Search' ,ascending=False)[['Cust_ID','Gender','Total Search']].head(10)
sns.barplot(data=plt_data,
           x='Cust_ID',
           y='Total Search',
           hue='Gender',
           order=plt_data.sort_values('Total Search',ascending=False).Cust_ID)
plt.title("Top 10 Cust_ID based on Total Searches")
plt.show()


# In[22]:


x=df.iloc[:,2:].values
x


# In[23]:


scale=MinMaxScaler()
features=scale.fit_transform(x)
features


# # Elbow method to get the optimal K value

# In[24]:


inertia=[]
for i in range(1,16):
    k_means=KMeans(n_clusters=i)
    k_means=k_means.fit(features)
    inertia.append(k_means.inertia_)


# In[25]:


#Elbow graph
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
plt.plot(range(1,16),inertia, 'bo-')
plt.xlabel('No of clusters'),plt.ylabel('Inertia')


# In[26]:


#Kelbow Visualizer 
plt.subplot(1,2,2)
kmeans=KMeans()
visualize=KElbowVisualizer(kmeans,k=(1,16))
visualize.fit(features)
plt.suptitle("Elbow Graph and Elbow Visualizer")
visualize.poof()
plt.show()


# # Silhouette Score for each k value

# In[27]:


silhouette_avg=[]
for i in range(2,16):
    #initialize kmeans cluster
    kmeans=KMeans(n_clusters=i)
    cluster_labels=kmeans.fit_predict(features)
    #Silhouette Score
    silhouette_avg.append(silhouette_score(features,cluster_labels))


# In[28]:


plt.figure(figsize=(10,7))
plt.plot(range(2,16),silhouette_avg, 'bX-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis for optimal K')


# # Kmeans Model Here we will take K value as 3 as per Elbow Method

# In[29]:


model=KMeans(n_clusters=3)
model=model.fit(features)


# In[30]:


y_km=model.predict(features)
centers=model.cluster_centers_


# In[31]:


df['Cluster']=pd.DataFrame(y_km)
df.to_csv("Cluster_data", index=False)


# In[32]:


df["Cluster"].value_counts()


# In[33]:


sns.countplot(data=df,x='Cluster')
plt.show()


# # Analyze the clusters

# In[34]:


c_df=pd.read_csv('Cluster_data')
c_df.head()


# In[35]:


c_df['Total Search']=c_df.iloc[:,3:38].sum(axis=1)


# # Analyze the cluster 0

# In[36]:


c1_0=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==0')
c1_0


# In[37]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(data=c_df.query('Cluster==0'),x='Gender')
plt.title('Customers count')

plt.subplot(1,2,2)
sns.barplot(data=c1_0,x='Gender' ,y='Total Search')
plt.title('Total Searches by Gender')
plt.suptitle('No. of customer and their total searches in "Cluster 0" ')
plt.show()


# # Analyse the cluster 1

# In[38]:


c1_1=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==1')
c1_1


# In[39]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(data=c_df.query('Cluster==1'),x='Gender')
plt.title('Customers count')

plt.subplot(1,2,2)
sns.barplot(data=c1_1,x='Gender' ,y='Total Search')
plt.title('Total Searches by Gender')
plt.suptitle('No. of customer and their total searches in "Cluster 1" ')
plt.show()


# # Analyze the cluster 2

# In[40]:


c1_2=c_df.groupby(['Cluster','Gender'], as_index=False).sum().query('Cluster==2')
c1_2


# In[41]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(data=c_df.query('Cluster==2'),x='Gender')
plt.title('Customers count')

plt.subplot(1,2,2)
sns.barplot(data=c1_2,x='Gender' ,y='Total Search')
plt.title('Total Searches by Gender')
plt.suptitle('No. of customer and their total searches in "Cluster 2" ')
plt.show()


# # Overall Analysis

# In[42]:


final_df=c_df.groupby(['Cluster'],as_index=False).sum()
final_df


# In[43]:


plt.figure(figsize=(15,6))
sns.countplot(data=c_df,x='Cluster',hue='Gender')
plt.title('Total Customer on each cluster')
plt.show()


# In[44]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.barplot(data=final_df,x='Cluster' ,y='Total Search')
plt.title('Total Searches by each group')

plt.subplot(1,2,2)
sns.barplot(data=final_df,x='Cluster' ,y='Orders')
plt.title('Past Orders by each group')
plt.suptitle('No. of times customer searched the products and their past orders')
plt.show()


# In[ ]:





# In[ ]:




