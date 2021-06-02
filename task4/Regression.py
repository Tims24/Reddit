#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[23]:


merged_data = pd.read_csv('merged_data.csv')


# In[24]:


merged_data.head()


# In[24]:


merged_data_x = merged_data.drop(['id', 'score', 'date'], axis=1)
merged_data_y = merged_data.score


# In[1]:


from sklearn.ensemble import RandomForestRegressor


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(merged_data_x, merged_data_y, test_size = 0.30, random_state = 42)


# In[2]:


clf_rf = RandomForestRegressor()


# In[4]:


from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


parametrs = {'n_estimators':  [ 40], 'max_depth': [12]}


# In[6]:


grid_clf_rf = GridSearchCV(clf_rf, param_grid=parametrs, cv=5)


# In[82]:


grid_clf_rf.fit(X_train, y_train)


# In[83]:


grid_clf_rf.best_params_


# In[84]:


best_grid_clf_rf = grid_clf_rf.best_estimator_


# In[85]:


best_grid_clf_rf.fit(X_train, y_train)


# In[86]:


feature_importances = best_grid_clf_rf.feature_importances_


# In[87]:


feature_importances_df = pd.DataFrame({'features': list(X_train),
                                      'feature_importances': feature_importances})


# In[88]:


imp = pd.DataFrame(best_grid_clf_rf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))


# In[89]:


best_grid_clf_rf.score(X_train, y_train)


# In[37]:


y_pred = best_grid_clf_rf.predict(X_test)


# In[90]:


best_grid_clf_rf.score(X_test, y_test)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(merged_data_x, merged_data_y, test_size = 0.20, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[32]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[33]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# # Рекомендации
# * Добавление новых признаков из текста постов с помощью инструментов NLP
# * Проверка гипотезы, что наличие определенных слов или отношение поста к какой-либо тематике влияет на рейтинг

# In[ ]:





# In[ ]:





# In[29]:


sns.kdeplot(data=merged_data, x='date')


# In[53]:


sns.kdeplot(data=merged_data, x='post_karma')


# In[ ]:





# In[52]:


sns.kdeplot(data=merged_data, x='comment_karma')


# In[48]:


sns.lineplot(x=merged_data.upvoted, y=merged_data.score)


# In[49]:


sns.lineplot(x=merged_data.post_karma, y=merged_data.score)


# In[51]:


sns.lineplot(x=merged_data.comment_karma, y=posts.comments_count)


# In[39]:


sns.lineplot(x=posts.comments_count, y=posts.score)


# In[34]:


sns.kdeplot(data=posts_data, x='comments_count')


# In[35]:


sns.kdeplot(data=posts_data, x='score')


# In[36]:


ax = sns.violinplot(y=posts_data['upvoted'])


# In[21]:


g = sns.heatmap(posts_data)


# In[ ]:




