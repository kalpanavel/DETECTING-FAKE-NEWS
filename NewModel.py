
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("Dataset/DataSet.csv")

print(df.shape)
df = df.dropna()
print(df.shape)

df_data = df[["text","label"]]
# Features and Labels
df_x = df_data['text']
df_y = df_data.label

corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(random_state=0)
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

clreport = classification_report(y_test, y_pred)

y_pred = classifier.predict(X_test)

print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of MLP Classifier')
plt.show()

# Creating a pickle file for the classifier
filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))