import math
import random
from collections import defaultdict
from pprint import pprint

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16,9)})
df = pd.read_csv('Combined_News_DJIA.csv', encoding='utf-8')
df.head()
train=df[df['Date']<'20150101']
test=df[df['Date']>'20141231']
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ", regex=True, inplace=True)
list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index
data.head(5)
for index in new_index:
    data[index]=data[index].str.lower()
data.head(1)
' '.join(str(x) for x in data.iloc[1,0:25])
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    headlines[0]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)
predictions=randomclassifier.predict(test_dataset)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)
