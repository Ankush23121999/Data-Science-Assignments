#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing requests to extract content from a url
import requests
# Beautifulsoup is for web scrapping...used to scrap specific content
from bs4 import BeautifulSoup as bs  
import re 
iphone_reviews=[]


# In[2]:


iphone_snapdeal=[]
url1 = "https://www.snapdeal.com/product/apple-iphone-5c-16-gb/988871559/reviews?page="
url2 = "&sortBy=RECENCY&vsrc=rcnt#defRevPDP"

for i in range(1,10):
  ip=[]  
  base_url = url1+str(i)+url2
  response = requests.get(base_url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  temp = soup.findAll("div",attrs={"class","user-review"})# Extracting the content under specific tags  
  for j in range(len(temp)):
    ip.append(temp[j].find("p").text)
  iphone_snapdeal=iphone_snapdeal+ip


# In[3]:


### Removing repeated reviews 
iphone_snapdeal = list(set(iphone_snapdeal))


# In[4]:


# Writing reviews into text file 
with open("ip_snapdeal.txt","w",encoding="utf-8") as snp:
    snp.write(str(iphone_snapdeal))


# In[5]:


iphone_snapdeal


# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import TweetTokenizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from matplotlib.pyplot import imread
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[7]:


get_ipython().system(' pip install textblob')


# In[9]:


get_ipython().system(' pip install wordcloud')


# In[14]:


ip_snapd=pd.read_csv('ip_snapdeal.txt',encoding='Latin-1')
ip_snapd


# In[15]:


# test processing
ip_snapd=' '.join(ip_snapd)
ip_snapd


# In[16]:


# Remove Punctuations 
no_punc_text=ip_snapd.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[17]:


import nltk
nltk.download('punkt')


# In[18]:


# remove https or url within text
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


# In[19]:


text_tokens=word_tokenize(no_url_text)
print(text_tokens)


# In[20]:


# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[21]:


# Tokens count
len(text_tokens)


# In[22]:


# Remove Stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[23]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[24]:


# Stemming (Optional)
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[25]:


# Lemmatization
import spacy
from spacy.lang.en.examples import sentences
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[26]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[27]:


clean_comments=' '.join(lemmas)
clean_comments


# In[28]:


# feature extraction
cv=CountVectorizer()
clean_commentscv=cv.fit_transform(lemmas)


# In[29]:


print(cv.vocabulary_)


# In[30]:


print(clean_commentscv.toarray()[100:200])


# In[31]:


print(clean_commentscv.toarray().shape)


# In[32]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[33]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[34]:


# tfidf vectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[35]:


# genertae wordcloud
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[36]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('pron')
stopwords.add('rt')
stopwords.add('yeah')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='white', max_words=100,colormap='Set1',stopwords=stopwords).generate(clean_comments)
# Plot
plot_cloud(wordcloud)


# In[37]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_comments
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[38]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[39]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[40]:


# Counting the noun & verb tokens
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
# viewing top ten results
wd_df[0:10]


# In[41]:


# from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(iphone_snapdeal))
sentences
from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(iphone_snapdeal))
sentences


# In[42]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[43]:


# Emotion Lexicon - Affin
affin=pd.read_csv("D:/ExcelR/Data Science Assignments/Assignments/Text Mining/Afinn.csv",sep=',',encoding='Latin-1')
affin


# In[44]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[45]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[46]:


# manual testing
calculate_sentiment(text='great')


# In[47]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[48]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[49]:


sent_df.sort_values(by='sentiment_value')


# In[50]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[51]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[52]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[53]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[54]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(20,8))
sns.distplot(sent_df['sentiment_value'],color="black")


# In[55]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(20,8))
sns.lineplot(y='sentiment_value',x='index',data=sent_df,color='red')


# In[56]:


# Correlation analysis
sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(20,8),title='Sentence sentiment value to sentence word count',color='red')

