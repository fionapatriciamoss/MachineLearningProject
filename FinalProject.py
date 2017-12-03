# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:33:29 2017
@author: fmoss1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from scipy.stats import itemfreq
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, roc_curve, auc, hamming_loss
from sklearn.cluster import DBSCAN, KMeans

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

dataset_week1 = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week1.csv')
dataset_week2 = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week2.csv')
dataset_week3 = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week3.csv')
dataset_week4 = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv')

#dataset_week_int = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week2.csv')

def preprocessing(dataset):
    new = dataset.drop(['Date first seen', 'Flows', 'Tos', 'attackID', 'attackDescription'], axis = 1)

    new['Bytes'] = np.where(new['Bytes'].str[-1].str.contains('M') == True, (new['Bytes'].str[:-1].astype(float))*1000000, new['Bytes'])
    
    #dataframes = [new]
    
    new['Src Pt'] = pd.cut(new['Src Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
    new['Dst Pt'] = pd.cut(new['Dst Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
    new['Src IP Addr'] = new['Src IP Addr'].str.split('_').str[0]
    new['Dst IP Addr'] = new['Dst IP Addr'].str.split('_').str[0]
    
    counts_src = new['Src IP Addr'].value_counts()
    counts_dst = new['Dst IP Addr'].value_counts()
    
    df = new[new['Src IP Addr'].isin(counts_src[counts_src > 1000].index)]
    df = new[new['Dst IP Addr'].isin(counts_dst[counts_dst > 1000].index)]
    
    
    df1 = new[new['Src IP Addr'].isin(counts_src[counts_src <= 1000].index)]
    df1 = df1.append(new[new['Dst IP Addr'].isin(counts_src[counts_src <= 1000].index)])
    
    df['Src IP Addr'] = df.groupby('Src IP Addr')['Src IP Addr'].transform('count')
    df['Dst IP Addr'] = df.groupby('Dst IP Addr')['Dst IP Addr'].transform('count')
    
    return df
    
#new = dataset_week2.drop(['Date first seen', 'Flows', 'Tos', 'attackID', 'attackDescription'], axis = 1)
#
#new['Bytes'] = np.where(new['Bytes'].str[-1].str.contains('M') == True, (new['Bytes'].str[:-1].astype(float))*1000000, new['Bytes'])
#
##dataframes = [new]
#
#new['Src Pt'] = pd.cut(new['Src Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
#new['Dst Pt'] = pd.cut(new['Dst Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['System', 'User', 'Dynamic'])
#new['Src IP Addr'] = new['Src IP Addr'].str.split('_').str[0]
#new['Dst IP Addr'] = new['Dst IP Addr'].str.split('_').str[0]
#
#counts_src = new['Src IP Addr'].value_counts()
#counts_dst = new['Dst IP Addr'].value_counts()
#
#df = new[new['Src IP Addr'].isin(counts_src[counts_src > 1000].index)]
#df = new[new['Dst IP Addr'].isin(counts_dst[counts_dst > 1000].index)]
#
#
#df1 = new[new['Src IP Addr'].isin(counts_src[counts_src <= 1000].index)]
#df1 = df1.append(new[new['Dst IP Addr'].isin(counts_src[counts_src <= 1000].index)])
#
#df['Src IP Addr'] = df.groupby('Src IP Addr')['Src IP Addr'].transform('count')
#df['Dst IP Addr'] = df.groupby('Dst IP Addr')['Dst IP Addr'].transform('count')


#new['Src IP Addr'].groupby('Src IP Addr').count()
#pd.value_counts(new['Src Pt'])

#

dataset_week1 = preprocessing(dataset_week1)
dataset_week2 = preprocessing(dataset_week2)
dataset_week3 = preprocessing(dataset_week3)
dataset_week4 = preprocessing(dataset_week4)

dataframes = [dataset_week1, dataset_week2, dataset_week3, dataset_week4]

result = pd.concat(dataframes)

result = result.values

result_labels = result[:, 9].astype('int32')
result_labels = result[:, 9]
result_features = result[:, 0:9]

#result_features_encoded = pd.get_dummies(data=result_features, columns=['Proto', 'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 'Flags'])
#result_labels_encoded = pd.get_dummies(result_labels)


label_En = LabelEncoder()
result[:, 1] = label_En.fit_transform(result[:, 1])
result[:, 2] = label_En.fit_transform(result[:, 2])
result[:, 3] = label_En.fit_transform(result[:, 3])
result[:, 4] = label_En.fit_transform(result[:, 4])
result[:, 5] = label_En.fit_transform(result[:, 5])
result[:, 8] = label_En.fit_transform(result[:, 8])
result[:, 9] = label_En.fit_transform(result[:, 9])
result[:, 10] = label_En.fit_transform(result[:, 10])
#result[:, 11] = label_En.fit_transform(result[:, 11])

one_hot_encoder = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 8])
result_features = one_hot_encoder.fit_transform(result_features).toarray()

#one_hot_encoder = OneHotEncoder(categorical_features = 'all')
#result_labels = one_hot_encoder.fit_transform(result_labels).toarray()

# =============================================================================
# MULTI-CLASS CLASSIFICATION
# =============================================================================

result_labels = label_binarize(result_labels, classes=[0,1,2,3,4])

# Splitting the data into training and testing samples and labels
sample_train, sample_test, label_train, label_test = train_test_split(result_features, result_labels, test_size=0.25, random_state=0) 

# Normalizing the data
scaler = StandardScaler().fit(sample_train)
sample_train = scaler.transform(sample_train) 
sample_test = scaler.transform(sample_test)

# MultiClass Logistic Regression (One-versus-All)
classif = OneVsRestClassifier(linear_model.LogisticRegression())
classif.fit(sample_train, label_train)
label_score = classif.decision_function(sample_test)
idxsort = np.argsort(label_score) 
label_pred = idxsort[:,-1]
print('\n one-vs-all: %.2f' % accuracy_score(label_test,label_pred, normalize = False))
a = confusion_matrix(label_test, label_pred)
#falsealarm = a[0][1] / (a[0][1]+a[1][1])
print (hamming_loss(label_test, label_pred))
print (a)
# False alarms for each class

plot_confusion_matrix(a, classes=['attacker', 'normal', 'suspicious', 'unknown', 'victim'])

print('\n one-vs-all: %.2f' % f1_score(label_test,label_pred, average='micro'))
print('\n one-vs-all: %.2f' % precision_score(label_test, label_pred, average='micro'))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
n_classes = 5
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_test[:, i], label_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
plt.figure()
for i in range(n_classes):
    
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
plt.show()









# approach 2: one-vs-one (review the approach)
classif = OneVsOneClassifier(linear_model.LogisticRegression())
classif.fit(sample_train, label_train)
# explain similar output and decision strategy 
label_score = classif.decision_function(sample_test)
idxsort = np.argsort(label_score) 
label_pred = idxsort[:,-1]
print('\n one-vs-one: %.2f' % accuracy_score(label_test,label_pred))


# approach 3: error-correcting output-code (review the approach)
classif = OutputCodeClassifier(linear_model.LogisticRegression(),code_size=1) # explain code_size is a relative size 
classif.fit(sample_train, label_train)
label_pred = classif.predict(sample_test)
print('\n ecoc: %.2f' % accuracy_score(label_test,label_pred))

plt.hist(label_train,alpha=0.3,ec='k')

def find_accuracy(number):
    accmc = np.zeros(number)
    for i in range(0,number):
        idxexam = np.where(label_test==(i+1))[0]
        accmc[i] = accuracy_score(label_test[idxexam],label_pred[idxexam])
    return accmc

idxlen = [6, 50, 6, 50, 50] # explain numbers 
idxtrain = np.empty(1)
for i in range(0,5):
    idxclass = np.where(label_train==(i+1))[0]
    idxtrain = np.r_[idxtrain, idxclass[0:idxlen[i]]]
idxtrain = np.delete(idxtrain,[0])
idxtrain = idxtrain.astype('uint8')
#
sample_train_sub = sample_train[idxtrain,:]
label_train_sub = label_train[idxtrain]
plt.hist(label_train_sub) # show more balanced distribution
#
classif = OneVsOneClassifier(linear_model.LogisticRegression())
classif.fit(sample_train_sub, label_train_sub)
label_score = classif.decision_function(sample_test)
idxsort = np.argsort(label_score) 
label_pred = idxsort[:,-1]+1
print('\n downsample: one-vs-one: %.2f' % accuracy_score(label_test,label_pred))

print (find_accuracy(5))  




















numcluster = 2
kmeans = KMeans(n_clusters=numcluster, random_state=0).fit(sample_train)
idxcluster = kmeans.labels_ 

#-- step 3: for each cluster, build an One-Class SVM 
clf = svm.OneClassSVM(kernel='linear')
# initialize prediction label array
label_pred = np.zeros((numcluster,len(label_test)))
# examine each cluster and learn an OCSVM
for i in range(0,numcluster):  
    clf.fit(sample_train[idxcluster==i,:])
    label_pred[i,:] = clf.predict(sample_test)

#-- step 4: different local OCSVMs label testing instances differently 
# need to aggregate their labeling results for one instance: here, we try majority voting 
label_pred_sum = np.sum(label_pred,axis=0)
label_pred_sum[label_pred_sum>=0] = 1
label_pred_sum[label_pred_sum<0] = -1
#
a = confusion_matrix(label_test, label_pred_sum.round())
precision = a[1][1] / (a[0][1]+a[1][1])
recall = a[1][1] / (a[1][0]+a[1][1])
falsealarm = a[0][1] / (a[0][1]+a[1][1])
print('\n Multi-Class OCSVM: Precision = %2.2f, Recall = %2.2f, FalseAlarm = %2.2f' % (precision,recall,falsealarm))














# step 2. perform multi-class learning approaches
# using binary relevance

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(sample_train, label_train[:, :5])

# predict
predictions = classifier.predict(sample_test)


# approach 1: one-vs-all (review the approach)
classif = OneVsRestClassifier(linear_model.LogisticRegression())
classif.fit(sample_train, label_train[:, :4])
# explain output label_score 
label_score = classif.decision_function(sample_test)
# explain decision strategy here
idxsort = np.argsort(label_score) 
label_pred = idxsort[:,-1]+1
print('\n one-vs-all: %.2f' % accuracy_score(label_test[:, :4],label_pred))


## approach 2: one-vs-one (review the approach)
#classif = OneVsOneClassifier(linear_model.LogisticRegression())
#classif.fit(sample_train, label_train)
## explain similar output and decision strategy 
#label_score = classif.decision_function(sample_test)
#idxsort = np.argsort(label_score) 
#label_pred = idxsort[:,-1]+1
#print('\n one-vs-one: %.2f' % accuracy_score(label_test,label_pred))

