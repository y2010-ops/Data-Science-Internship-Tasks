# Data-Science-Internship-Tasks
#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics internship

# ## Task1 : Prediction using Supervised ML

# ### Demonstrated by : Yamini Vijaywargiya 

# #### Problem: What will be predicted score if a student studies for 9.25 hrs/ day?

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


X = "http://bit.ly/w-data"
s_data = pd.read_csv(X)
print("Data imported Sucessfully")

s_data.head(15)


# In[11]:


s_data.plot(x='Hours' , y='Scores' , style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[12]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 


# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### TRAINING PART

# In[15]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[16]:



line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### Making predictions

# In[17]:


print(X_test) 
y_pred = regressor.predict(X_test)


# In[18]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### Result

# In[23]:


# dataset = np.array(9.25)
dataset = dataset.reshape(-1,1)
pred = regressor.predict(dataset)
print("if the student for 9.25 hr/day, the score is {}.".format(pred))


# In[22]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

