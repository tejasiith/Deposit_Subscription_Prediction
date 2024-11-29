#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ## Read the dataset

# In[3]:


data = pd.read_csv(r'C:\Users\DELL\Documents\Finlatics_Project\DsResearch\Banking\banking_data.csv')


# In[4]:


data


# In[5]:


data.head()


# In[7]:


data.shape


# In[8]:


data.describe


# In[9]:


data.info()


# ## Handling Misiing values

# In[10]:


# Check for missing values
data.isnull().sum()


# In[11]:


# Fill missing values with appropriate values
data = data.fillna({'job': 'unknown', 'education': 'unknown'})


# In[12]:


data.isnull().sum()


# ## Encode Categorical Features

# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


# Encode categorical features
le = LabelEncoder()
data['job'] = le.fit_transform(data['job'])
data['marital'] = le.fit_transform(data['marital'])
data['education'] = le.fit_transform(data['education'])
data['default'] = le.fit_transform(data['default'])
data['housing'] = le.fit_transform(data['housing'])
data['loan'] = le.fit_transform(data['loan'])
data['contact'] = le.fit_transform(data['contact'])
data['month'] = le.fit_transform(data['month'])
data['poutcome'] = le.fit_transform(data['poutcome'])
data['y'] = le.fit_transform(data['y'])


# In[33]:


# Encode categorical features
categorical_cols = ['job', 'marital', 'education','marital_status', 'default', 'housing', 'loan', 'contact', 'month','day_month', 'poutcome']
le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])


# In[40]:


X_train.dtypes


# ## Split Data into Features and Target

# In[34]:


X = data.drop('y', axis=1)
y = data['y']


# ## Split Data into Train and Test Sets

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train a Random Forest Classifier model

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# ## Evaluate the model

# In[41]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[43]:


y_pred = rf_model.predict(X_test)


# ### Accuracy

# In[44]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# ### Precision

# In[45]:


precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")


# ### Recall

# In[46]:


recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")


# ### F1_score

# In[48]:


f1 = f1_score(y_test, y_pred)


# In[49]:


print(f"F1-score: {f1:.2f}")


# ## Questions and Answers

# In[52]:


# Set the style for seaborn
sns.set(style="whitegrid")


# In[53]:


# 1. Distribution of age among the clients
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Distribution of Age Among Clients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[54]:


# 2. Job type distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='job', order=data['job'].value_counts().index)
plt.title('Job Type Distribution Among Clients')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


# In[55]:


# 3. Marital status distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='marital')
plt.title('Marital Status Distribution Among Clients')
plt.ylabel('Count')
plt.show()


# In[56]:


# 4. Level of education distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='education')
plt.title('Level of Education Among Clients')
plt.ylabel('Count')
plt.show()


# In[57]:


# 5. Proportion of clients with credit in default
default_proportion = data['default'].value_counts(normalize=True)
plt.figure(figsize=(8, 5))
sns.barplot(x=default_proportion.index, y=default_proportion.values)
plt.title('Proportion of Clients with Credit in Default')
plt.ylabel('Proportion')
plt.show()


# In[58]:


# 6. Distribution of average yearly balance
plt.figure(figsize=(10, 6))
sns.histplot(data['balance'], bins=30, kde=True)
plt.title('Distribution of Average Yearly Balance Among Clients')
plt.xlabel('Average Yearly Balance')
plt.ylabel('Frequency')
plt.show()


# In[59]:


# 7. Number of clients with housing loans
housing_loan_count = data['housing'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=housing_loan_count.index, y=housing_loan_count.values)
plt.title('Number of Clients with Housing Loans')
plt.ylabel('Count')
plt.show()


# In[60]:


# 8. Number of clients with personal loans
personal_loan_count = data['loan'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=personal_loan_count.index, y=personal_loan_count.values)
plt.title('Number of Clients with Personal Loans')
plt.ylabel('Count')
plt.show()


# In[61]:


# 9. Communication types used for contacting clients
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='contact')
plt.title('Communication Types Used for Contacting Clients')
plt.ylabel('Count')
plt.show()


# In[62]:


# 10. Distribution of the last contact day of the month
plt.figure(figsize=(10, 6))
sns.histplot(data['day'], bins=31, kde=False)
plt.title('Distribution of Last Contact Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Frequency')
plt.show()


# In[63]:


# 11. Last contact month distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='month', order=data['month'].value_counts().index)
plt.title('Last Contact Month Distribution')
plt.ylabel('Count')
plt.show()


# In[64]:


# 12. Distribution of the duration of the last contact
plt.figure(figsize=(10, 6))
sns.histplot(data['duration'], bins=30, kde=True)
plt.title('Distribution of Duration of Last Contact')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()


# In[65]:


# 13. Number of contacts performed during the campaign for each client
plt.figure(figsize=(10, 6))
sns.histplot(data['campaign'], bins=10, kde=False)
plt.title('Number of Contacts Performed During the Campaign')
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.show()


# In[66]:


# 14. Distribution of days passed since last contacted from a previous campaign
plt.figure(figsize=(10, 6))
sns.histplot(data['pdays'], bins=30, kde=False)
plt.title('Distribution of Days Passed Since Last Contacted from Previous Campaign')
plt.xlabel('Days Passed')
plt.ylabel('Frequency')
plt.show()


# In[67]:


# 15. Number of contacts performed before the current campaign
plt.figure(figsize=(10, 6))
sns.histplot(data['previous'], bins=10, kde=False)
plt.title('Number of Contacts Performed Before the Current Campaign')
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.show()


# In[68]:


# 16. Outcomes of previous marketing campaigns
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='poutcome')
plt.title('Outcomes of Previous Marketing Campaigns')
plt.ylabel('Count')
plt.show()


# In[69]:


# 17. Distribution of clients who subscribed to a term deposit vs. those who did not
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='y')
plt.title('Clients Subscribed to a Term Deposit vs. Not Subscribed')
plt.ylabel('Count')
plt.show()


# In[70]:


# 18. Correlation between different attributes and the likelihood of subscribing to a term deposit
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Between Attributes')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




