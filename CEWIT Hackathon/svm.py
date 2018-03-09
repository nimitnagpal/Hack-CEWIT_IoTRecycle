import numpy as np
from sklearn import preprocessing, cross_validation, neighbors , svm
import pandas as pd

df=pd.read_csv('data.csv')
df.replace('?',-9999, inplace=True)
X = np.array(df.drop(['Metal'],1))
y = np.array(df['Metal'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.3)

clf = svm.SVC()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

example_measures = np.array([[12],[75],[140]])
example_measures = example_measures.reshape(3,-1)
predictions = clf.predict(example_measures)
print(accuracy)
print(predictions)
