# Libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import preprocess

path = 'data/youtube_spam/youtube_spam_dataset.csv'
df = pd.read_csv(path)

def buildModel():
    mnb = MultinomialNB()
    df['processed_text'] = df['CONTENT'].apply(preprocess.textPreprocess)
    countVectorizer = CountVectorizer(min_df=0.001,max_df=.10)
    vectors = countVectorizer.fit_transform(df['processed_text']).toarray()
    X_train, X_test, y_train, y_test = train_test_split(vectors, df['CLASS'].values, test_size=.2, random_state=1234)
    mnb.fit(X_train, y_train)
    df['prediction'] = mnb.predict(vectors)
    df.to_csv('prediction.csv')
    print('model successfully built')
    pred_train = mnb.predict(X_train)
    pred_test = mnb.predict(X_test)
    print()
    print('Train classification report :',classification_report(pred_train, y_train))
    print('Test classification report :',classification_report(pred_test, y_test))
    print()
    pickle.dump(mnb,open('pickles/spam_model.pkl', 'wb'))
    pickle.dump(countVectorizer,open('pickles/vectorizer.pkl', 'wb'))
    print('pickle files successfully saved in pickles')

def detectSpam(text):
    model = pickle.load(open('pickles/spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('pickles/vectorizer.pkl', 'rb'))
    text = preprocess.textPreprocess(text)
    text = vectorizer.transform([text])
    return model.predict(text)

if __name__=='__main__':
    x = input('enter your comment : ')
    print(x)
    print(detectSpam(x))