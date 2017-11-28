# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:33:29 2017

@author: fmoss1
"""

import pandas as pd
import numpy as np

from scipy.stats import itemfreq
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#dataset_week1 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/WISENT-CIDDS-001/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week1.csv')
dataset_week2 = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week2.csv')
#dataset_week3 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/WISENT-CIDDS-001/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week3.csv')
#dataset_week4 = pd.read_csv('C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/WISENT-CIDDS-001/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv')

dataset_week_int = pd.read_csv('C:/Users/Fiona/Downloads/Semester-3/Machine Learning/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week2.csv')
new = dataset_week2.drop(['Date first seen', 'Flows', 'Tos', 'attackDescription'], axis = 1)

new['Bytes'] = np.where(new['Bytes'].str[-1].str.contains('M') == True, (new['Bytes'].str[:-1].astype(float))*1000000, new['Bytes'])

dataframes = [new]



new['Src Pt'] = pd.cut(new['Src Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['0', '1', '2'])
new['Dst Pt'] = pd.cut(new['Dst Pt'], bins=[0, 1023, 49151, 65535], include_lowest=True, labels=['0', '1', '2'])



#new['Src IP Addr'].groupby('Src IP Addr').count()
#pd.value_counts(new['Src Pt'])

result = pd.concat(dataframes)

result = result.values

result_labels = result.iloc[:, 9:12]
result_features = result.iloc[:, 0:9]

result_features_encoded = pd.get_dummies(data=result_features, columns=['Proto', 'Src IP Addr', 'Src Pt', 'Dst IP Addr', 'Dst Pt', 'Flags'])
result_labels_encoded = pd.get_dummies(result_labels)


label_En = LabelEncoder()
result[:, 1] = label_En.fit_transform(result[:, 1])
result[:, 2] = label_En.fit_transform(result[:, 2])
result[:, 3] = label_En.fit_transform(result[:, 3])
result[:, 4] = label_En.fit_transform(result[:, 4])
result[:, 5] = label_En.fit_transform(result[:, 5])
result[:, 8] = label_En.fit_transform(result[:, 8])
result[:, 9] = label_En.fit_transform(result[:, 9])
result[:, 10] = label_En.fit_transform(result[:, 10])
result[:, 11] = label_En.fit_transform(result[:, 11])

one_hot_encoder = OneHotEncoder(categorical_features = [1, 2, 3, 4, 5, 8])
result_features = one_hot_encoder.fit_transform(result_features).toarray()