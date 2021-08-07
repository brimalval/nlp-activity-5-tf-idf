#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from num2words import num2words
import math

import os
from os.path import isfile, join
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests

import matplotlib.pyplot as plt


# In[22]:


# Artificial intelligence URLs
source_urls = [
     "http://realpeoplepress.com/blog/",
     "https://en.wikipedia.org/wiki/Natural_language_processing",
     "https://www.ibm.com/cloud/learn/what-is-artificial-intelligence",
     "https://www.sas.com/en_us/insights/analytics/what-is-artificial-intelligence.html",
     "https://plato.stanford.edu/entries/artificial-intelligence/",
     "https://www.nist.gov/artificial-intelligence",
     "https://edgy.app/examples-of-artificial-intelligence",
     "https://www.datamation.com/artificial-intelligence/what-is-artificial-intelligence/",
     "https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_overview.htm",
     "https://www.mygreatlearning.com/blog/what-is-artificial-intelligence/"
]

# Software Engineering URLs
#source_urls = [
#    "https://www.unr.edu/cse/undergraduates/prospective-students/what-is-software-engineering",
#    "https://www.thoughtco.com/what-is-software-engineering-958652",
#    "https://careerkarma.com/careers/software-engineer/",
#    "https://www.guru99.com/what-is-software-engineering.html",
#    "https://www.computerscience.org/careers/software-engineer/",
#    "https://www.careerexplorer.com/careers/software-engineer/",
#    "https://www.geeksforgeeks.org/software-engineering/",
#    "https://www.indeed.com/career-advice/careers/what-does-a-software-engineer-do",
#    "https://www.thebalancecareers.com/software-engineer-skills-list-2062483",
#    "https://www.snhu.edu/online-degrees/bachelors/bs-in-computer-science/software-engineering"
#]


# In[23]:


sources = []
for url in source_urls:
    print()
    print("Attempting to retrieve", url, "...")
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    print("Done retrieving", url)
    sources.append(soup.get_text())


# In[4]:


# Cleaning functions
def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer = SnowballStemmer("english")
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def lemmatize(data, tag=None):
    data = str(data)
    lemma = nltk.stem.WordNetLemmatizer()
    if tag != None:
        return lemma.lemmatize(data, tag)
    return lemma.lemmatize(data)

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, '-', ' ')
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = lemmatize(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = lemmatize(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    return data


# In[5]:


def freq_doc_and_total(data):
    word_counts = []
    word_totals = []
    word_maps = []
    for document in data:
        word_total = 0
        words_in_doc = dict()
        if type(document) == str:
            document = word_tokenize(document)
        for word in document:
            if word not in words_in_doc:
                words_in_doc[word] = 0
            words_in_doc[word] += 1
            word_total += 1
        word_totals.append(word_total)
        counts_arr = words_in_doc.items()
        # Sort counts by freq. (index 1) in descending order
        counts_sorted = sorted(counts_arr, key=lambda x:x[1], reverse=True)
        word_counts.append(counts_sorted)
        word_maps.append(words_in_doc)
    return (word_counts, word_totals, word_maps)

def calculate_tf(sorted_word_count, total):
    all_tf = dict()
    for word, freq in sorted_word_count:
        all_tf[word] = freq/total
    
    return all_tf

def calculate_idf(word_maps, total_docs, sorted_total_counts):
    all_idf = dict()
    i = 0
    for word_count in sorted_total_counts:
        for word, freq in word_count:
            occurrences_across_docs = 0
            if word in all_idf:
                continue
            for word_map in word_maps:
                if word in word_map:
                    occurrences_across_docs += 1
            # Log base 10 of total documents over a word's occurrence across the docs
            idf = math.log(total_docs/occurrences_across_docs, 10)
            all_idf[word] = idf
    return all_idf
        
                
    
def calculate_tf_idf(data):
    document_count = len(data)
    word_counts, word_totals, word_maps = freq_doc_and_total(data)
    tf = []
    for i in range(document_count):
        tf.append(calculate_tf(word_counts[i], word_totals[i]))
    idf = calculate_idf(word_maps, document_count, word_counts)
    
    tf_idfs = []
    for doc in tf:
        doc_tf_idf = dict()
        for word, word_tf in doc.items():
            tf_idf = word_tf * idf[word]
            doc_tf_idf[word] = tf_idf
        tf_idfs.append(doc_tf_idf)
    return tf, idf, tf_idfs
    
        


# In[24]:


processed_text = []

for source in sources:
    processed_text.append(str(preprocess(source)))

tf, idf, tf_idfs = calculate_tf_idf(processed_text)
compiled_tf_idfs = {}
for doc_tf_idf in tf_idfs:
    for word, tf_idf in doc_tf_idf.items():
        if word not in compiled_tf_idfs:
            compiled_tf_idfs[word] = 0
        compiled_tf_idfs[word] += tf_idf

all_tf_idf_array = compiled_tf_idfs.items()
all_tf_idf_array = sorted(all_tf_idf_array, key=lambda x:x[1], reverse=True)
# print(compiled_tf_idfs['nlp'])
# entities_df = pd.DataFrame(all_tf_idf_array)
# entities_df.columns = ["Word", "Score"]
# entities_df

fig, ax = plt.subplots()

# S L I C E
words_to_graph = 10
words = [x[0] for x in all_tf_idf_array[:words_to_graph]]
y_pos = np.arange(len(words))
scores = [x[1] for x in all_tf_idf_array[:words_to_graph]]

ax.barh(y_pos, scores, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(words)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('TF-IDF Score')
ax.set_title('TF-IDF Scores for Different Terms')

plt.show()


# In[ ]:




