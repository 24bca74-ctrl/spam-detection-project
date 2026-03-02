import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label']=data['label'].map({'ham':0,'spam':1})

X_train,X_test,Y_train,Y_test = train_test_split(
data['message'],
data['label'],
test_size=0.2)

vectorizer = CountVectorizer()

X_train_count = vectorizer.fit_transform(X_train)

model = MultinomialNB()

model.fit(X_train_count,Y_train)

# Save files
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model Created Successfully")
print("Model Accuracy:", model.score(vectorizer.transform(X_test), Y_test))