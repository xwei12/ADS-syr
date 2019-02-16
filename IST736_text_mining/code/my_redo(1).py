# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 02:49:24 2018

@author: killu
"""

# -*- coding: utf-8 -*-
import os
path = r"C:\Users\killu\Desktop\ist736\project"
os.chdir(path)
import pandas
import random
random.seed(2018)
import nltk
from nltk import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

text = pandas.read_csv('textdata.csv')
testdata = pandas.read_csv('testdata.csv')
author_text = text[['authorname', 'content']]
train = author_text
import re
for document in train.content:
    document=re.sub(r'[0-9]','@',document)
    
    

X_train = train['content']
y_train = train['authorname']
test = testdata[['authorname', 'content']]
for document in test.content:
    document=re.sub(r'[0-9]','@',document)
X_test = test['content']
y_test = test['authorname']
import scipy
from sklearn.neighbors import KNeighborsClassifier


unigram_count = CountVectorizer(encoding='latin-1',lowercase=False, binary=False, min_df=8)
unigram_tfidf = TfidfVectorizer(encoding='latin-1',lowercase=False, use_idf=True, min_df=8)
gram12_count = CountVectorizer(encoding='latin-1',lowercase=False, ngram_range=(1,2), min_df=8)
gram12_tfidf = TfidfVectorizer(encoding='latin-1',lowercase=False, ngram_range=(1,2), use_idf=True, min_df=8)
target_names=text.authorname.unique()
index=0;
from sklearn.tree import DecisionTreeClassifier
def printReport(vectorizer):
    global index
    print(vectorizer.__repr__)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec =  vectorizer.transform(X_test)
#    csvname1="train"+str(index)+".csv"
#    csvname2="test"+str(index)+".csv"
    #np.savetxt(csvname1, X_train_vec, delimiter=',')
    #np.savetxt(csvname2, X_test_vec, delimiter=',')
#    nb_clf= MultinomialNB()
#    nb_clf.fit(X_train_vec,y_train)
#    print("nb_clf score",nb_clf.score(X_test_vec,y_test)) #0.6656
#    y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
#    #print(precision_score(y_test, y_pred, average=None))
#    #print(recall_score(y_test, y_pred, average=None))
#    cm=confusion_matrix(y_test, y_pred)    
#    print(classification_report(y_test, y_pred, target_names=target_names))
#    
    
    svm_clf=LinearSVC(C=1)
    svm_clf.fit(X_train_vec,y_train)
    print("tree score:",svm_clf.score(X_test_vec,y_test)) #0.6812
    y_pred_svm = svm_clf.predict(X_test_vec)
    print("y_pred_svm:",y_pred_svm)
    print("ytest:",y_test)
    cmsvmsss=confusion_matrix(y_test, y_pred_svm)
    print(precision_score(y_test, y_pred_svm, average=None))
    print(recall_score(y_test, y_pred_svm, average=None))
    print("cm:\n",cmsvmsss)
    print(classification_report(y_test, y_pred_svm, target_names=target_names))
    plt.figure(figsize=(30,25))
    ax=plt.subplot(111)
    sns.heatmap(cmsvmsss,ax=ax, cmap="YlGnBu")
    index+=1
   # numpy.savetxt("foo.csv", cmsvmsss, delimiter=",")

    
#printReport(unigram_count)
#printReport(unigram_tfidf)
#printReport(gram12_count)
printReport(gram12_tfidf)

#heat map

