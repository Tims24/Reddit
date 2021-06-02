#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


posts = pd.read_csv('export/posts.csv')
authors = pd.read_csv('export/authors.csv')


# In[3]:


posts['posted_at_timestamp'] = posts['posted_at'].apply(lambda x: pd.Timestamp(x))
posts['posted_at_timestamp'] = posts.posted_at_timestamp.astype('int64') // 10**9
posts['date'] = pd.to_datetime(posts.posted_at_timestamp, unit='s')


# In[4]:


posts.head()


# In[5]:


posts.astype


# In[6]:


posts.describe()


# In[7]:


posts['space_count'] = posts.body.str.count("\n")


# In[8]:


posts['len_text'] = posts.body.str.len()


# In[9]:


posts.head()


# In[10]:


sns.kdeplot(data=posts, x='date')


# Видно, что наибольшее количество постов наблюдается в промежуток с 2014 по 2016 год.
# В связи с этим можно провести анализ, и посмотреть, что хранится в постах за 2018-2020 годы

# In[11]:


import nltk


# In[12]:


import requests

from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from joblib import Parallel, delayed
import spacy
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'lemmatizer'])

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

def preprocess_parallel(texts, chunksize=100):
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

def lemmatize_pipe(doc):
    text = doc.text.lower()
    return normalize_text(text)

def read_stopwords(path):
    url = str(path)
    r = requests.get(url)
    return set(r.text.split('\n'))

def normalize_tokens(tokens):
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(' '.join(tokens))]

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok
              for tok in tokens
              if tok not in stopwords and len(tok) >= min_length]
    return tokens

def normalize_text(text):
    words = [w for sent in sent_tokenize(text.lower())
             for w in regexp_tokenize(sent, r'(?u)\b\w{4,}\b')]
    words = normalize_tokens(words)
    words = remove_stopwords(words, stopwords)
    return ' '.join(words)

lemmatizer = WordNetLemmatizer()

stopwords = set(stopwords.words('english')) | read_stopwords('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt')


# In[13]:


# posts['normalized'] = posts['body'].apply(lambda text: normalize_text(text))
posts['normalized'] = preprocess_parallel(posts['body'], chunksize=1000)


# In[14]:


posts.to_csv('export/posts_normalized.csv', encoding='utf-8', index=False)


# In[15]:


import datetime

def datestring_to_timestamp(datestring):
    return datetime.datetime.strptime(datestring, '%d.%m.%Y').timestamp()

def posts_by_year(df, year):
    return df[(df['posted_at_timestamp'] >= datestring_to_timestamp(f'01.01.{year}')) & (df['posted_at_timestamp'] < datestring_to_timestamp(f'31.12.{year}'))]


# In[16]:


posts2018 = posts_by_year(posts, 2018)
posts2019 = posts_by_year(posts, 2019)
posts2020 = posts_by_year(posts, 2020)


# In[17]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_word_cloud(text, title, picture_fn='out.png', **wc_kwargs):
    wc = WordCloud(**wc_kwargs).generate(text)
    print(title)
    plt.figure(figsize=(12,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(picture_fn)


# Наиболее популярные слова за все время

# In[18]:


plot_word_cloud(" ".join(posts['normalized']), 'All', 'export/cloud.png', max_words=100, background_color='black')


# 2018 год

# In[19]:


plot_word_cloud(" ".join(posts2018['normalized']), '2018', 'export/cloud2018.png', max_words=100, background_color='black')


# 2019 год

# In[20]:


plot_word_cloud(" ".join(posts2019['normalized']), '2019', 'export/cloud2019.png', max_words=100, background_color='black')


# 2020 год

# In[21]:


plot_word_cloud(" ".join(posts2020['normalized']), '2020', 'export/cloud2020.png', max_words=100, background_color='black')


# # Рекомендации
# * Посмотреть как на рейтинг статьи влияет наличие определнных слов
# * Подобрать группы слов, которые бы относились к определенным темам

# In[22]:


posts_data = posts.drop(['id', 'title', 'body', 'normalized', 'url', 'posted_at', 'created_at', 'updated_at', 'short_url', 'reddit_id', 'subreddit_id'], axis=1)
posts_data = posts_data.fillna(0)
posts_data['author_id'] = posts_data['author_id'].apply(lambda id: int(id))
posts_data.fillna({'author_id': 0.0})
posts_data = posts_data.rename(columns={'author_id': 'id'})

authors_data = authors.drop(['name', 'created_at', 'updated_at', 'registered_at', 'reddit_id'], axis=1)


# In[23]:


merged_data = pd.merge(posts_data, authors_data, on='id')


# In[24]:


merged_data


# In[25]:


merged_data_x = merged_data.drop(['id', 'score', 'date'], axis=1)
merged_data_y = merged_data.score


# In[26]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(merged_data_x, merged_data_y, test_size = 0.30, random_state = 42)
clf_rf = RandomForestRegressor()
parametrs = {'n_estimators':  [ 40], 'max_depth': [12]}
grid_clf_rf = GridSearchCV(clf_rf, param_grid=parametrs, cv=5)


# In[27]:


grid_clf_rf.fit(X_train, y_train)


# In[28]:


grid_clf_rf.best_params_


# In[29]:


best_grid_clf_rf = grid_clf_rf.best_estimator_


# In[30]:


best_grid_clf_rf.fit(X_train, y_train)


# In[31]:


feature_importances = best_grid_clf_rf.feature_importances_


# In[32]:


feature_importances_df = pd.DataFrame({'features': list(X_train),
                                      'feature_importances': feature_importances})


# In[33]:


imp = pd.DataFrame(best_grid_clf_rf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))


# In[34]:


from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from ast import literal_eval
import pyLDAvis.gensim_models

docs = [[token for token in text.split()] for text in posts['normalized']]
dictionary = Dictionary(docs)

# Drop too common and too rare words
dictionary.filter_extremes(no_below=20, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Set training parameters.
num_topics = 15
chunksize = 2000
passes = 20
iterations = 400

temp = dictionary[0] 
id2word = dictionary.id2token 

model = LdaModel(
    corpus=corpus,
    id2word=id2word, 
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=None
)


# In[35]:


pyLDAvis.enable_notebook()
ldavis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
ldavis


# In[36]:


coherence_model_lda = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[37]:


pyLDAvis.save_html(ldavis, 'export/topic_modeling.html')

