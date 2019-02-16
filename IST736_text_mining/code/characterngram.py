# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 04:08:25 2018

@author: killu
"""

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

text = pandas.read_csv('textdata.csv')
testdata = pandas.read_csv('testdata.csv')
author_text = text[['authorname', 'content']]
train = author_text
X_train = train['content']
y_train = train['authorname']
test = testdata[['authorname', 'content']]
X_test = test['content']
y_test = test['authorname']
import scipy
import re
def word2ngrams(text, n=3, exact=True):
    """ Convert text into character ngrams. """
    text=re.sub(r'[0-9]','@',text)
    #text=re.sub(r'[0-9]','@',text).lower()
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

#train_vec=pd.DataFrame(index=range(2500))
#index=0
#for document in X_train:
#    for ngram in word2ngrams(document):
#        if ngram not in train_vec.columns:
#            train_vec[ngram]=pd.Series(0,index=range(2500))
#            train_vec[ngram][index]=1
#        else:
#            train_vec[ngram][index]+=1
#    print(index)
#    index+=1
#
#test_vec=train_vec.copy()
#test_vec[:]=0
#index=0
#for document in X_test:
#    for ngram in word2ngrams(document):
#        if ngram in test_vec.columns:
#            test_vec[ngram][index]+=1
#    print(index)
#    index+=1
#test_vec.to_pickle("test_3grams.pkl")
import pickle
from sklearn.neighbors import KNeighborsClassifier
#train_vec=pickle.load(open("train_char3grams.pkl","rb"))
#test_vec=pickle.load(open("test_char3grams.pkl","rb"))
from sklearn import tree

nb_clf=LinearSVC(C=1)
y_pred_char=nb_clf.fit(train_vec,y_train).predict(test_vec)
target_names=text.authorname.unique()
print("nb_clf score",nb_clf.score(test_vec,y_test))
print(precision_score(y_test, y_pred_char, average=None))

#print(classification_report(y_test, y_pred_char, target_names=target_names))
#nb_clf= MultinomialNB()
#y_pred_char=nb_clf.fit(train_vec,y_train).predict(test_vec)
#target_names=text.authorname.unique()
#print("nb_clf score",nb_clf.score(test_vec,y_test))
#print(precision_score(y_test, y_pred_char, average=None))
#
#print(classification_report(y_test, y_pred_char, target_names=target_names))
#for ngram in train_vec.columns:
#    ngram.replace("\r\n","**")
    
    
#unigram_count = CountVectorizer(encoding='latin-1', binary=False, min_df=8)
#unigram_tfidf = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=8)
#gram12_count = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=8)
#gram12_tfidf = TfidfVectorizer(encoding='latin-1', ngram_range=(1,2), use_idf=True, min_df=8)
#target_names=text.authorname.unique()
#index=0;
#def printReport(vectorizer):
#    global index
#    print(vectorizer.__repr__)
#    X_train_vec = vectorizer.fit_transform(X_train)
#    X_test_vec =  vectorizer.transform(X_test)
##    csvname1="train"+str(index)+".csv"
##    csvname2="test"+str(index)+".csv"
#    #np.savetxt(csvname1, X_train_vec, delimiter=',')
#    #np.savetxt(csvname2, X_test_vec, delimiter=',')
#    nb_clf= MultinomialNB()
#    nb_clf.fit(X_train_vec,y_train)
#    print("nb_clf score",nb_clf.score(X_test_vec,y_test)) #0.6656
#    y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
#    print(precision_score(y_test, y_pred, average=None))
#    print(recall_score(y_test, y_pred, average=None))
#    cm=confusion_matrix(y_test, y_pred)    
#    print(classification_report(y_test, y_pred, target_names=target_names))
#    
#    svm_clf = LinearSVC(C=1)
#    svm_clf.fit(X_train_vec,y_train)
#    print("svm score:",svm_clf.score(X_test_vec,y_test)) #0.6812
#    y_pred_svm = svm_clf.predict(X_test_vec)
#    cmsvm=confusion_matrix(y_test, y_pred_svm)
#    print(precision_score(y_test, y_pred_svm, average=None))
#    print(recall_score(y_test, y_pred_svm, average=None))
#    print(classification_report(y_test, y_pred_svm, target_names=target_names))
#    index+=1
#    
#    
#printReport(unigram_count)
#printReport(unigram_tfidf)
#printReport(gram12_count)
#printReport(gram12_tfidf)
