import os
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors , svm
import pandas as pd

df=pd.read_csv('data.csv')
df.replace('?',-9999, inplace=True)
X = np.array(df.drop(['Metal'],1))
y = np.array(df['Metal'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.3)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
a = str(accuracy)
example_measures = np.array([[12],[75],[140]])
example_measures = example_measures.reshape(3,-1)
predictions = clf.predict(example_measures)
np.savetxt("output1.txt",predictions, newline=",")
file = open("output.txt","w")
file.write(a)
file.close()
y=np.loadtxt("output1.txt",delimiter=",",usecols=(0,2),unpack=True)
fout=open("result.php","w")
out = 0
for i in y:
    if(i==1):
        out += i*5
    elif(i==2):
        out += i*6
    else:
        out += i*7
fout.write(str(out))
fout.close()
print(accuracy)
print(predictions)
#out = (a*5)+(b*4)+(c*3)

        
#np.savetxt("output.txt",(accuracy,predictions), fmt="%d")

