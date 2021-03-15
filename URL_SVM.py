from numpy import array
import numpy as np

file=open("../urldata/1k_samples_rows.csv","r")
X= []
label=[]
# preprocessting
for line in file:
    Features=[]
    # splitting the line based on symbol
    number_strings =line.split(',') 
    Features=number_strings[1:]
    label.append(float(number_strings[0].strip()))  
    # Convert floating numbers to integers
    numbers = [float(n) for n in Features] 
    X.append(numbers)
     
X=array(X)
label=array(label)
Y=label
# Splitting the dataset into train and test set with the ratio of 80:20
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train=array(X_train)
X_test=array(X_test)
Y_train=array(Y_train)
Y_test=array(Y_test)

#calculate the time
import time
start_time = time.time()

from sklearn import svm
SVM_model = svm.SVC(gamma='scale',  probability=True)

# training the model
SVM_model.fit(X_train,Y_train)

# Making the predictions based on the test data
SVM_predicted_values = SVM_model.predict(X_test)
SVM_predicted_proba = SVM_model.predict_proba(X_test)

# printing the model training time
print("--- %s seconds ---" % (time.time() - start_time))

# Accuracy Calculation
from sklearn.metrics import accuracy_score
accscore = accuracy_score(Y_test,SVM_predicted_values)
print ("Accuracy is %s" % accuracy_score(Y_test,SVM_predicted_values))

# Precision Calculation
from sklearn.metrics import precision_score
prscore = precision_score(Y_test,SVM_predicted_values, average='weighted')
print("Precision Score is %s" % prscore)

# Recall Calculation
from sklearn.metrics import recall_score
rescore = recall_score(Y_test,SVM_predicted_values, average='weighted')
print("Recall Score is %s" % rescore)

# F1-score calculation
from sklearn.metrics import f1_score
f1score = f1_score(Y_test,SVM_predicted_values, average='weighted')
print("F-1 Score is %s" % f1score)

from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve
#keeping probabilities for true prediction
fpr, tpr, thresholds = roc_curve(Y_test,SVM_predicted_proba[:,1])
auc = roc_auc_score(Y_test,SVM_predicted_values)
precision, recall, thresholds = precision_recall_curve(Y_test,SVM_predicted_proba[:,1])

import matplotlib.pyplot as plt
plt.figure(num=1,figsize=(11,5))
plt.plot(['Accuracy','Precision','Recall','F-1 Score'],np.array([accscore,prscore,rescore,f1score])*100,'ro--')
plt.figure(num=2,figsize=(11,5))
plt.plot(fpr,tpr,'go--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.figure(num=3,figsize=(11,5))
plt.plot(recall,precision,'^k--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.figure(num=3,figsize=(11,5))
plt.show()

