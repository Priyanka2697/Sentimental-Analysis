#!/usr/bin/env python
# coding: utf-8

# ## Extract information from any e-book (PDF) to get Summarization and Sentiment Analysis

# #  Installing and importing the libraries 

# In[1]:


#install the packages

#!pip install --upgrade pip
#!pip install PyPDF2
#!pip install nltk
#!pip install -U spacy
#!python -m spacy download en_core_web_lg


# In[2]:


#import the libraries

import PyPDF2 as pdf #used for text entraction from pdf
import nltk #nlp framework
#nltk.download('all') #need to only install/download once. this is installing nltk data (all)

import re #can use regex functions
import spacy #open source library for advanced nlp
from spacy.lang.en.stop_words import STOP_WORDS #spacy.lang.en is an english language class in spacy. Library for STOP_WORDS package
from string import punctuation #Library for punctuation package


# In[3]:


# read the input pdf file

##file = open(r"C:\Users\prathikm\Documents\Project_101\ebooks\Harry_Potter_Book1_and_The_Philosophers_Stone.pdf", 'rb')
file = open(r"C:\Users\Babu\Documents\P-109\ebook\Harry_Potter_Book1_and_The_Philosophers_Stone.pdf",'rb')
pdf_reader = pdf.PdfFileReader(file)

# from pages 0 to end of pages (numpages), get/read the page and extract it text with appending it to last extracted page. Then print it.

text=''
for i in range(0,pdf_reader.numPages):
    pageObj = pdf_reader.getPage(i)
    text=text+pageObj.extractText()
print(text)


# In[7]:


type(text)


# # Cleaning & formatting of extracted text for further NLP analysis

# In[8]:


# text cleaning using tokenization - method used is .split() 
# make a list from the above string 'text' such that- each list is not made by one whitespace (which is by default ' '), but each string is defined only
# if there are 2 whitespace characters '  '
# why? so that any new paragraphs or text starting from a new page is taken as a new list, not individual words

new_text = text.split('  ')
#new_text = [line for line in text.split('\n') if line.strip() != '']
new_text


# In[9]:


# now we have a list of sentences, just remove the \n and replace it with single space, \n\n and replace it with since space ' ' and so on. 
# this changes as per input book. For example - if input book is a fiction, we will have these characters below which needs to be converted to single whitespace
# but if the book is a maths book for example, then there might be some other unwanted formulas which needs to be converted to single whitespace

list2 = [x.replace('\n', ' ').replace('\n\n', ' ').replace(' -- ',' ').replace('- ',' ') for x in new_text]


# In[10]:


# all the \n, \n\n, space -- space and - space in the above list have been converted to single whitespace list.

list2


# In[11]:


# the datatype is still list

type(list2)


# In[12]:


# we need to work with this list and thus for NLP pre-processing, we will have to convert it to a string

# LTS Function  - to convert list to string
def listToString(s): 
    # initialize an empty string
    str1 = "" 
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    # return string  
    return str1 
        
        
# Driver code/ pass out list to this LTS function to get output as sting
s = list2
now_string = listToString(s)
print(now_string) 


# In[13]:


# check our output of function

now_string


# In[14]:


#is our output string? yes

type(now_string)


# #  Generate the Summary of all input text

# In[15]:


# WHY summarization? Picking only important information for complete information, selection process easier, also increases chance to process multiple books
# at once.


# In[16]:


stop_words = list(STOP_WORDS) #pre-defined stop words in package STOP_WORDS
stop_words


# In[17]:


nlp = spacy.load('en_core_web_lg') #NLP Trained Models & Pipelines in Spacy
#nlp.max_length = 1230000 # or even higher for book 4
nlp.max_length = 1638156 # or even higher for book 5


# In[18]:


doc = nlp(now_string) # "nlp" Object is used to create documents with linguistic annotations 'tokenize the input string'. 


# In[19]:


type(doc)


# In[20]:


tokens = [token.text for token in doc] # build list of token words. NOTE: punctuation and stop words are also part of original tokens
print(tokens)


# In[21]:


punctuation = punctuation + '\n' #check punctuation (for removing) and add new line \n in punctuation as it is not there in it by default
punctuation


# In[22]:


word_frequencies = {} #calculate word frequencies from doc
for word in doc:
    if word.text.lower() not in stop_words: #other than stopwords, punctuation and if it is new word -> add as new word count, else -> old word + 1
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


# In[23]:


word_frequencies


# In[24]:


from operator import itemgetter
sorted(word_frequencies.items(), key=itemgetter(1), reverse = True)


# In[25]:


max_frequency = max(word_frequencies.values()) #get maximum of above frequency


# In[26]:


max_frequency #max frequency of a word in the above is 2270.


# In[27]:


for word in word_frequencies.keys(): #divide each word by this max frequency to get normalized frequency of words. 2270/2270 = 1, that word has 1 as normalized frequency which is max.
    word_frequencies[word] = word_frequencies[word]/max_frequency


# In[28]:


print(word_frequencies) #print the normalized frequency


# In[29]:


sentence_tokens = [sent for sent in doc.sents] #do the sentence tokenization
print(sentence_tokens)


# In[30]:


sentence_scores = {} #calculate the sentences score. Calculate most important sentence based on normalized frequency and word frequency
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]


# In[31]:


sentence_scores #each sentence is scored


# In[32]:


from heapq import nlargest #task is to get 30% of sentences with maximum score from above. this gives important sentences only.


# In[33]:


select_length = int(len(sentence_tokens)*0.3) #calculate 30% of total sentences
select_length


# In[34]:


summary = nlargest(select_length, sentence_scores,key = sentence_scores.get) #select 1887 (30%) of sentences with maximum count/score. nlargest(n,iterable, keys)


# In[35]:


summary #these sentences represent the summary of the text, based on max importance of each senetence. 1887 (30%) most important sentences


# In[36]:


final_summary = [word.text for word in summary] #generate list of summary


# In[37]:


final_summary


# In[38]:


summary_2 = ''.join(final_summary) #join the summary list to make a paragraph
print(summary_2)


# In[39]:


len(text) #length of original text characters


# In[40]:


len(summary_2) #length of summary text characters


# In[41]:


summary_2


# In[42]:


summary_2.count('Snape')


# In[43]:


percentage_of_text_in_summary = (len(summary_2)/len(text))*100
print("Percentage of text in final summary is :", percentage_of_text_in_summary)


# # Part 4 - EDA 

# In[50]:


conda install -c conda-forge wordcloud


# In[51]:


import pandas as pd
#from imblearn.over_sampling import SMOTE

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter
from wordcloud import WordCloud
from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier


from keras import models
from keras import layers
import keras
from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding


# In[52]:


hp = pd.read_csv(r'C:\Users\Babu\Desktop/hp_script.csv',encoding='cp1252')
hp


# In[53]:


hp.head()


# In[54]:


hp['character_name'].value_counts()


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(30,24))
sns.countplot(y='character_name', data=hp, order=hp.character_name.value_counts().iloc[:20].index, palette="tab10")
plt.xlabel('Number of lines of dialogue', fontsize=30)
plt.ylabel('Character', fontsize=40)
plt.title('Character Importance by Number of Lines of Dialogue', fontsize=40)
plt.show()


# In[57]:


import warnings
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[59]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('hp')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(summary_2)
# Plot
plot_cloud(wordcloud)


# In[ ]:





#  ## Sentiment Analysis

# In[44]:


type(final_summary) #type of input is list


# In[45]:


final_summary[0:10] #get first 10 items/sentences in list


# # Model Building - VaderSentiment

# In[46]:


get_ipython().system('pip install vaderSentiment')


# In[47]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[48]:


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


# In[49]:


import pandas as pd

scores =[]
#sentences = ["A really bad, horrible book.","A good, awesome, wonderful, cool book !!!  :)"]

for sentence in final_summary:
    score = analyser.polarity_scores(sentence)
    scores.append(score)
    
#Converting List of Dictionaries into Dataframe
dataFrame= pd.DataFrame(scores)

print("Sentiment Score for each sentence in the book (Summarized i.e. Important sentences) :-\n")
print(dataFrame)

print("Overall Sentiment Score for complete book :-\n",dataFrame.mean())


# In[ ]:




