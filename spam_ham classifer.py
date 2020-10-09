# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:41:59 2020

@author: prasa
"""


import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords



#import the file and here label and message is separeted with tab so i have given as \t 
messages = pd.read_csv('D:/Work space/smsspamcollection/SMSSpamCollection', sep='\t', names=["labels", "Text_Message"])


#Now we have to preprocess the text by stemming or lemmitization and join all them into a corpus

sentenece_lemt = WordNetLemmatizer()
portstem = PorterStemmer()
corpus = []


for i in range(0, len(messages)):
    convert_reg = re.sub('[^a-zA-Z]', ' ', messages['Text_Message'][i])
    convert_lower = convert_reg.lower()
    convert_split = convert_lower.split()
    convert_split=[sentenece_lemt.lemmatize(word) for word in convert_split if not word in stopwords.words('english')]
    convert_split = ' '.join(convert_split)
    corpus.append(convert_split)

from sklearn.feature_extraction.text import CountVectorizer
count_vect= CountVectorizer(max_features=3000)    
X= count_vect.fit_transform(corpus).toarray()





#now taking y (dependent variable) by using dummy variable as we have only two labels

y= pd.get_dummies(messages['labels'])
y=y.iloc[:,1].values

#splititing train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split( X, y, test_size=.20, random_state=0)


#now training the model with navie bais algo
from sklearn.naive_bayes import MultinomialNB
spam_ham_detect_model=MultinomialNB().fit(X_train,y_train)


#predicting after training the model
y_pred = spam_ham_detect_model.predict(X_test)


#y_predict = y_pred.predict(count_vect.transform(['FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv']))
#y_predict = spam_ham_detect_model.predict(count_vect.transform(['nah think go usf life around though']))


from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test,y_pred)

#now we have to find accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)



    
    
    
    
