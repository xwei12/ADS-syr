# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:45:23 2018

@author: killu
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
text = pd.read_csv('textdata.csv')
testdata = pd.read_csv('testdata.csv')
author7=testdata.loc[testdata['authorname'] == testdata['authorname'].unique()[18]]
d = pd.DataFrame(np.zeros((50,50)))
for authorIndex1 in range(50):
    author7=testdata.loc[testdata['authorname'] == testdata['authorname'].unique()[authorIndex1]]
    for authorIndex2 in range(50):
        authorna=testdata['authorname'].unique()[authorIndex2]
        author13=testdata.loc[testdata['authorname'] == authorna]
        gram12_tfidf = TfidfVectorizer(encoding='latin-1',lowercase=False, ngram_range=(1,2), use_idf=True, min_df=8)
        pd.options.mode.chained_assignment = None
        import re
        author7.content.replace(r'[0-9]','@',inplace=True, regex=True)
        author13.content.replace(r'[0-9]','@',inplace=True, regex=True)
        vectorizer = gram12_tfidf
        X = vectorizer.fit_transform(author7.content)
        indices1 = sum(X).data
        features = vectorizer.get_feature_names()
        top_n = indices1.size
        top_features1 = [features[i] for i in range(top_n)]
        dict1={}
        range1=range(indices1.size)
        for i in range1:
            dict1[top_features1[i]]=indices1[i]
        #print("indices sizes:",indices1.size)
        #print(top_features1," for author 7")
        
        X = vectorizer.fit_transform(author13.content)
        indices2 = sum(X).data
        #print("indices sizes:",indices2.size)
        features = vectorizer.get_feature_names()
        top_n = indices2.size
        top_features2 = [features[i] for i in range(top_n)]
        dict2={}
        range2=range(indices2.size)
        for i in range2:
            dict2[top_features2[i]]=indices2[i]
        
        #print(number2)
        #print(top_features2," for author 13")
        
        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3
         
        # Driver Code
        
        #print("in common",intersection(top_features1,top_features2))
        print(authorIndex1+1," and ",authorIndex2+1," in common ",len(intersection(top_features1,top_features2)))
        d.iloc[authorIndex1,authorIndex2]=len(intersection(top_features1,top_features2))
        
        import operator
        sorted_dict1 = sorted(dict1.items(), key=operator.itemgetter(1))
        sorted_dict2 = sorted(dict2.items(), key=operator.itemgetter(1))
        dict1_sorted=sorted_dict1[::-1]
        dict2_sorted=sorted_dict2[::-1]