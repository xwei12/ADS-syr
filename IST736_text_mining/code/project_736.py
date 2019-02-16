# -*- coding: utf-8 -*-
import pandas
import nltk
from nltk import *

text = pandas.read_csv('D:/COURSE/IST736/Project/textdata.csv')
author_text = text[['authorname', 'content']]
author_text.to_csv('D:/COURSE/IST736/Project/author-text.csv',index = False)

grouped = author_text['content'].groupby(author_text['authorname'])
glist = list(grouped)
groupedtext = []
authorlist = []
for au in range(len(glist)):
    author = glist[au][0]
    content = str(glist[au][1].values)
    authorlist.append([author])
    groupedtext.append([content])
author_text = {'author':authorlist, 'content': groupedtext}
df = pandas.DataFrame(author_text)                      
author_text = df

def alpha_filter(w):
    pattern = re.compile('^[^a-z]+$')
    if(pattern.match(w)):
        return True
    else:
        return False

wordslist = []   
for i in range(len(author_text)):
    unigram = []
    bigram = []
    author = str(author_text['author'][i])
    text = str(author_text['content'][i])
    words = nltk.word_tokenize(text)
    Lowerwords = [w.lower() for w in words]
    alphawords = [w for w in Lowerwords if w.isalpha()]
    totalwords = len(alphawords)
    unique = set(alphawords)
    uniquewords = len(unique)
    stopwords = nltk.corpus.stopwords.words('english')
    stoppedwords = [w for w in alphawords if (w not in stopwords)]
    fdist=FreqDist(stoppedwords)
    topkeys = fdist.most_common(20)
    for pair in topkeys:
        unigram.append(pair)
    wordsbigrams = list(nltk.bigrams(words))
    bigram_measures = nltk.collocations.BigramAssocMeasures
    finder = BigramCollocationFinder.from_words(Lowerwords)
    finder.apply_word_filter(alpha_filter)
    finder.apply_word_filter(lambda w: w in stopwords)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    for bscore in scored[:20]:
        bigram.append(bscore)
    wordslist.append([author, totalwords, uniquewords, unigram, bigram])
    
wordsdf = pandas.DataFrame(wordslist, columns = ['author', 'totalwords', 'uniquewords', 'unigram', 'bigram'])
wordsdf.to_csv('D:/COURSE/IST736/Project/author-words.csv',index = False)




























