# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:58:37 2018

@author: killu
"""

import os
cwd = os.getcwd()
path = r"C:\Users\killu\Desktop\ist736\project\C50train"
os.chdir(path)
import pandas as pd
from pathlib import PurePath
#create empty data frame
df1 = pd.DataFrame(columns=["authorname","filename","fullpath","content"])
index=0
for path, subFolders, files in os.walk(path):
        for file in files:
            nm, ext = os.path.splitext(file)
            if ext.lower().endswith(("txt")): #this is so i filter only what I want
                filepaths = os.path.join(os.path.abspath(path),file)
                p=PurePath(filepaths)
                #author label
                author = p.parts[-2]
                #read content
                txtfile=open(filepaths,"r")
                content=txtfile.read()
                txtfile.close()
                df1.loc[index]=[author,file,filepaths,content]    
                
                index = index+1
#save into csv file
#df1.to_csv("textdata.csv", sep=',', encoding='utf-8')