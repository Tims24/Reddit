#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


from datetime import datetime
from gensim.models import Word2Vec


# In[3]:


posts = pd.read_csv('posts.csv')
authors = pd.read_csv('authors.csv')


# # Выбор пространства признаков

# В первую очередь перевед дату в timestamp, чтобы в дальнешем модель могла работать с этими данными

# In[4]:


posts['posted_at_timestamp'] = posts['posted_at'].apply(lambda x: pd.Timestamp(x))
posts['posted_at_timestamp'] = posts.posted_at_timestamp.astype('int64') // 10**9


# In[5]:


posts.head()


# # Подсчет статистических параметров

# In[7]:


posts.describe()


# Найдем длину каждого поста, а также количество отступов(абзацев). Возможно рейтинг статьи как то будет коррелировать с этим признаком, т.к. наличие отступов задает читаемость поста

# In[5]:


posts['space_count'] = posts.body.str.count("\n")


# In[6]:


posts['len_text'] = posts.body.str.len()


# In[10]:


posts.head()


# In[7]:


model = Word2Vec(posts['body'].str.split(), min_count = 1, workers = 3, window = 3, sg = 1)


# # Частотный анализ коллекции

# In[5]:


import os
import requests
from operator import attrgetter
from pathlib import Path
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
import pymorphy2
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def plot_word_cloud(text, picture_fn='out.png', stopwords=None,
                    normalize=True, regexp=r'(?u)\b\w{4,}\b', **wc_kwargs):
    words = [w for sent in sent_tokenize(text)
             for w in regexp_tokenize(sent, regexp)]
    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)
    wc = WordCloud(**wc_kwargs).generate(' '.join(words))
    plt.figure(figsize=(12,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(picture_fn)

def normalize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(tok)[0].normal_form for tok in tokens]

def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok
              for tok in tokens
              if tok not in stopwords and len(tok) >= min_length]
    return tokens

def read_stopwords(path):
    url = str(path)
    r = requests.get(url)
    return set(r.text.split('\n'))

text = " ".join(posts['body'])

stopwords = read_stopwords('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt')

plot_word_cloud(text, 'out.png', stopwords=stopwords, max_words=100, background_color='black', normalize=True)


# По итогу анализа видно, какие слова наиболее часто встречаются в постах

# Также есть возможность посмотреть как эти слова похожи. Из такого анализа можно выявить связь между определнными темами

# In[13]:


model.most_similar('wife')[:10]


# In[12]:


model.most_similar('husband')[:10]


# In[16]:


model.most_similar('break-up')[:10]


# In[25]:


model.similarity('breakup', 'cheat')


# In[20]:


model.similarity('break-up', 'divorce')


# In[24]:


model.similarity('breakup', 'raped')


# In[29]:


model.most_similar('relationship')[:20]


# In[7]:


posts_data = posts.drop(['id', 'title', 'body', 'url', 'posted_at', 'created_at', 'updated_at', 'short_url', 'reddit_id', 'subreddit_id'], axis=1)


# In[8]:


posts_data = posts_data.fillna(0)


# In[9]:


posts_data['author_id'] = posts_data['author_id'].apply(lambda id: int(id))


# In[12]:


authors.head()


# In[14]:


authors_data = authors.drop(['name', 'created_at', 'updated_at', 'registered_at', 'reddit_id'], axis=1)


# In[15]:


posts_data.fillna({'author_id': 0.0})


# In[16]:


authors_data.head()


# In[17]:


posts_data = posts_data.rename(columns={'author_id': 'id'})


# In[18]:


posts_data.head()


# In[19]:


merged_data = pd.merge(posts_data, authors_data, on='id')


# In[20]:


merged_data['date'] = pd.to_datetime(merged_data.posted_at_timestamp, unit='s')


# In[21]:


merged_data


# # Построение распределения и анализ переменных

# In[22]:


merged_data.to_csv('merged_data.csv', encoding='utf-8', index=False)


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


# Посмотрим на значимость признаков

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


# # Поиск выбросов и аномалий

# In[29]:


sns.kdeplot(data=merged_data, x='date')


# In[53]:


sns.kdeplot(data=merged_data, x='post_karma')


# In[ ]:





# In[52]:


sns.kdeplot(data=merged_data, x='comment_karma')


# Видна зависимость рейтинга постов от признака 'upvoted'

# In[48]:


sns.lineplot(x=merged_data.upvoted, y=merged_data.score)


# In[49]:


sns.lineplot(x=merged_data.post_karma, y=merged_data.score)


# In[51]:


sns.lineplot(x=merged_data.comment_karma, y=posts.comments_count)


# In[39]:


sns.lineplot(x=posts.comments_count, y=posts.score)


# Как мы видим, большинство постов с низким рейтингом и количеством комментариев 

# In[34]:


sns.kdeplot(data=posts_data, x='comments_count')


# In[35]:


sns.kdeplot(data=posts_data, x='score')


# In[36]:


ax = sns.violinplot(y=posts_data['upvoted'])


# In[21]:


g = sns.heatmap(posts_data)


# In[23]:


sns.pairplot(posts, hue='subreddit_id')


# На графике ниже видно, что наиболее интересные посты, т.е. с высоким рейтингом у сообществ с id 1,2

# In[30]:


sns.lineplot(x='upvoted', y='score', hue='subreddit_id',sizes=(.25, 2.5), data=posts)


# In[85]:


special_words = ['sex', 'drugs', 'abuse', 'virgin', 'breakup', 'break-up', 'harassment', 'rap', 'gay', 'care', 'date']


# In[83]:


body = posts.body


# In[84]:


body.head()

