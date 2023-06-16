# import tensorflow as tf
# from keras import layers, optimizers, losses
# from keras.layers import Dense, Dropout
# import keras
# from keras.models import Sequential
# from keras import layers
# import matplotlib.pyplot as plt
# from tensorflow.keras.utils import to_categorical
# from tensorflow import keras
# import re
# import scipy.io
import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from keras.layers.convolutional import Conv2D
# from keras.layers import Dense
# from keras.layers.convolutional import MaxPooling2D,Conv1D,MaxPooling1D
# from keras.layers.core import Flatten
# from tensorflow.keras.layers import BatchNormalization
# from keras.layers import Activation
# from sklearn.neighbors import KNeighborsClassifier
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import keras

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-poster')

# print(tf.__version__)
# print(keras.__version__)



################################# Healthy


Data_0A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-0A.xlsx')
# print(Data_0A_H.head())

Data_1A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-1A.xlsx')
# print(Data_1A_H.head())

Data_2A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-2A.xlsx')
# print(Data_2A_H.head())

Data_3A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-3A.xlsx')
# print(Data_3A_H.head())

Data_4A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-4A.xlsx')
# print(Data_4A_H.head())

Data_5A_H= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_Healthy-5A.xlsx')
# print(Data_5A_H.head())




################################# morakab
Data_0A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-0A.xlsx')
# print(Data_0A_M.head())

Data_1A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-1A.xlsx')
# print(Data_1A_M.head())

Data_2A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-2A.xlsx')
# print(Data_2A_M.head())

Data_3A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-3A.xlsx')
# print(Data_3A_M.head())

Data_4A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-4A.xlsx')
# print(Data_4A_M.head())

Data_5A_M= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_morakab50%-5A.xlsx')
# print(Data_5A_M.head())




################################# Dynamic
Data_0A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-0A.xlsx')
# print(Data_0A_D.head())

Data_1A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-1A.xlsx')
# print(Data_1A_D.head())

Data_2A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-2A.xlsx')
# print(Data_2A_D.head())

Data_3A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-3A.xlsx')
# print(Data_3A_D.head())

Data_4A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-4A.xlsx')
# print(Data_4A_D.head())

Data_5A_D= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_50%Dynamic-5A.xlsx')
# print(Data_5A_D.head())





################################# static
Data_0A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-0A.xlsx')
# print(Data_0A_S.head())

Data_1A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-1A.xlsx')
# print(Data_1A_S.head())

Data_2A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-2A.xlsx')
# print(Data_2A_S.head())

Data_3A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-3A.xlsx')
# print(Data_3A_S.head())

Data_4A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-4A.xlsx')
# print(Data_4A_S.head())

Data_5A_S= pd.read_excel(r'./feature_ex_spilit125/Feature_ex_static50%-5A.xlsx')
# print(Data_5A_S.head())


# concat all imported data
Data_0A = pd.concat([Data_0A_H,Data_0A_M,Data_0A_D,Data_0A_S],ignore_index=True)
Data_1A = pd.concat([Data_1A_H,Data_1A_M,Data_1A_D,Data_1A_S],ignore_index=True)
Data_2A = pd.concat([Data_2A_H,Data_2A_M,Data_2A_D,Data_2A_S],ignore_index=True)
Data_3A = pd.concat([Data_3A_H,Data_3A_M,Data_3A_D,Data_3A_S],ignore_index=True)
Data_4A = pd.concat([Data_4A_H,Data_4A_M,Data_4A_D,Data_4A_S],ignore_index=True)
Data_5A = pd.concat([Data_5A_H,Data_5A_M,Data_5A_D,Data_5A_S],ignore_index=True)
# label is 0 for healthy, 1 for morakab, 2 for dynamic, 3 for static
# the length of each data is 125
# therefore the first 125 rows are 0, the second 125 rows are 1, the third 125 rows are 2, the fourth 125 rows are 3
# type is int
label = np.concatenate((np.zeros(125),np.ones(125),2*np.ones(125),3*np.ones(125)),axis=0).astype(int)
# add label to each data
Data_0A["label"] = label
Data_1A["label"] = label
Data_2A["label"] = label
Data_3A["label"] = label
Data_4A["label"] = label
Data_5A["label"] = label




## First Scenario: 3A as test data, others as training data
#  concat all except Data_3A to form training data
# Data_train = pd.concat([Data_0A,Data_1A,Data_2A,Data_4A,Data_5A],ignore_index=True)
# Data_train = Data_train.sample(frac=1).reset_index(drop=True) # shuffle the training data
# Data_test = Data_3A.copy()

# only use 4 features: "RMS_Va_0_H_S", "kurtosis_Va_0_H_S", "BeforeTMH_Va_0_H_S", "Mean_Frequency" and covert it to numpy array
# X_train = Data_train[["RMS_Va_0_H_S", "kurtosis_Va_0_H_S", "BeforeTMH_Va_0_H_S", "Mean_Frequency"]].to_numpy()
# Y_train = Data_train["label"].to_numpy()
# X_test = Data_test[["RMS_Va_0_H_S", "kurtosis_Va_0_H_S", "BeforeTMH_Va_0_H_S", "Mean_Frequency"]].to_numpy()
# Y_test = Data_test["label"].to_numpy()





## Second Scenario: randomly select 15% of each data as test data, others as training data
Data = pd.concat([Data_0A,Data_1A,Data_2A,Data_3A,Data_4A,Data_5A],ignore_index=True)
X = Data[["RMS_Va_0_H_S", "kurtosis_Va_0_H_S", "Med_Frequency", "Mean_Frequency"]].to_numpy()
Y = Data["label"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.15, random_state=42 , shuffle=True)



net_SVC = SVC(C=1.0, kernel='rbf', gamma='auto')
net_SVC.fit(X_train,Y_train)
print("net_SVC accuracy is {} on Train Dataset".format(net_SVC.score(X_train,Y_train)))
print("net_SVC accuracy is {} on Test Dataset".format(net_SVC.score(X_test,Y_test)))


predictions = net_SVC.predict(X_test)
cm = confusion_matrix(Y_test, predictions, labels=net_SVC.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=net_SVC.classes_)
disp.plot()
plt.show()

X = Data[["RMS_Va_0_H_S", "kurtosis_Va_0_H_S", "Crest_factor_Va_0_H_S" ,"Skewness_factor_Va_0_H_S" ,
          "BeforeTMH_Va_0_H_S" , "AfterTMH_Va_0_H_S" , "BeforeFMH_Va_0_H_S" , "AfterFMH_Va_0_H_S" , 
          "PeakOf_TMF_Va_0_H_S" , "PeakOf_FMF_Va_0_H_S" , "Mean_Frequency" , "Med_Frequency" , "Peak_To_Peak"]].to_numpy()
Y = Data["label"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.15, random_state=42 , shuffle=True)



from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X_train,Y_train)
#Calculating Prediction
y_predict_model = model.predict(X_test)
y_predict_model
 


#Calculating Details
print('model Train Score is : ' , model.score(X_train, Y_train))
print('model Test Score is : ' , model.score(X_test, Y_test))



#Calculating Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
confusion_matrix=confusion_matrix(Y_test,y_predict_model)
confusion_matrix



My_Data=Data.drop(['Unnamed: 0', 'label'], axis=1)
#X_valid_2=Data.drop(columns, axis=1)
My_Data.columns
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=My_Data.columns)
feat_importances

feat_importances = feat_importances.sort_values()
feat_importances

plt.figure(figsize=(10,10))
feat_importances.nlargest(30).plot(kind='barh')
plt.show()