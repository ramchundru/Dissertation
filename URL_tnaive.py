from numpy import array
import numpy as np

file=open("../urldata/1k_samples_rows.csv","r")
X= []
label=[]
for line in file:
    URLFeatures=[]
    # split the line
    number_strings = line.split(',')
    # Consider all features except the first one
    URLFeatures = number_strings[1:]  
    label.append(float(number_strings[0].strip()))
    numbers = [float(n) for n in URLFeatures] 
    # X is the feature and Y is the label 
    X.append(numbers) 

Y=label
# split data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)

#calculating time
import time
start_time = time.time()

# Applying Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
naive_model=GaussianNB()

# Training the model here.
naive_model.fit(X_train,Y_train)


# Making the predictions on test data 
naive_prediction=naive_model.predict(X_test)
naive_predicted_proba = naive_model.predict_proba(X_test)

#print("--- %s seconds ---" % (time.time() - start_time))
from sklearn.metrics import accuracy_score
accscore = accuracy_score(Y_test,naive_prediction)
print ("Accuracy is %s" % accuracy_score(Y_test,naive_prediction))

from sklearn.metrics import precision_score
precisionmetric = precision_score(Y_test,naive_prediction, average='weighted')
print("Precision Score is %s" % precisionmetric)

from sklearn.metrics import recall_score
recallmetric = recall_score(Y_test,naive_prediction, average='weighted')
print("Recall Score is %s" % recallmetric)

from sklearn.metrics import f1_score
f1metric = f1_score(Y_test,naive_prediction, average='weighted')
print("F-1 Score is %s" % f1metric)

print("--- %s seconds ---" % (time.time() - start_time))


from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve
#keeping probabilities for true prediction
fpr, tpr, thresholds = roc_curve(Y_test,naive_predicted_proba[:,1])
auc = roc_auc_score(Y_test,naive_prediction)
precision, recall, thresholds = precision_recall_curve(Y_test,naive_predicted_proba[:,1])

import matplotlib.pyplot as plt
plt.figure(num=1,figsize=(11,5))
plt.plot(['Accuracy','Precision','Recall','F-1 Score'],np.array([accscore,precisionmetric,recallmetric,f1metric])*100,'ro--')
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
