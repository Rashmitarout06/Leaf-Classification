# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:39:24 2016

@author: Rashmita Rout & Partha S Satpathy
"""
###################################IMPORT PACKAGES#############################
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

import cv2

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
###############################################################################

################################IMPORT TRAIN &TEST FILES#######################

#first we must begin with importing the train and test dataset
train = pd.read_csv("C:/Users/parth/Desktop/MIA/MidTerm/train.csv")
test = pd.read_csv("C:/Users/parth/Desktop/MIA/MidTerm/test.csv")

#converting the test and train datasets into dataframes
train_df= pd.DataFrame(train)
test_df= pd.DataFrame(test)

#we check both the datasets to find the columns present in them 
#and then find the number of missing values in each column
train_df.info()
test_df.info()
###############################################################################

###############################PLAYING WITH THE IMAGES#########################
##########################INCREASED OUR SCORE##################################
## We are storing the area, Perimeter, Asp Ratio and Equivalent Diameter
## information of all the images
## And adding them to the train and test features

##Adding four dummy columns to train and test
train_df['Area'] = train_df['id']
train_df['Perimeter'] = train_df['id']
train_df['AspRatio'] = train_df['id']
train_df['EquiDiam'] = train_df['id']

test_df['Area'] = test_df['id']
test_df['Perimeter'] = test_df['id']
test_df['AspRatio'] = test_df['id']
test_df['EquiDiam'] = test_df['id']

## This will loop through all the images and extrat the 
##area, perimeter, aspect ratio and Diameter of all images
for i in range(1,1584):
    #Path variable has the path for all 1584 images
    path = 'C:/Users/parth/Desktop/MIA/MidTerm/Data/images/'+str(i)+'.jpg'
    #Read the image details and store the details
    img = cv2.imread(path,0)
    ret,thresh = cv2.threshold(img,127,255,0)
    im,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    
    cnt = contours[0]
    M = cv2.moments(cnt)
   
    ##Calculate Area and Perimeter
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    
    ##Calculate Aspect Ratio
    x,y,w,h = cv2.boundingRect(cnt)
    asp_ratio = float(w)/h
    
    ##Calculate Equivalent Diameter
    equiDiam = np.sqrt(4*area/np.pi)
    
    ##Add the calculated values to train if the id in train matches the image file number
    for j in range(0,990):
        if(train_df.iat[j,194] == i):
            train_df.iat[j,194] = area
            train_df.iat[j,195] = perimeter
            train_df.iat[j,196] = asp_ratio
            train_df.iat[j,197] = equiDiam
    ##Add the calculated values to test if the id in test matches the image file number
    for k in range(0,594):
        if(test_df.iat[k,193] == i):
            test_df.iat[k,193] = area
            test_df.iat[k,194] = perimeter
            test_df.iat[k,195] = asp_ratio
            test_df.iat[k,196] = equiDiam

###############################################################################

###################CREATE TRAIN & TEST FEATURES AND LABELS#####################
#In Python we need to divide the data set into features and labels
#First, we will divide the train data to features and labels
#The train features will only have the predictor columns,i.e., 
#Removing the unnecessary ID and the dependent variable Species
train_features = train_df.drop(['id', 'species'], axis=1)
#train_features.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/train_features_1.csv",index=False)
#Saving the dependent variable to train labels
train_labels = train_df.pop('species')
#train_labels.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/train_labels_1.csv",index=False)
#Saving the test features
test_features = test_df.drop(['id'], axis=1)
#test_features.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/test_features_1.csv",index=False)
#Saving the test ids for the submission file
x_testid = test_df.pop('id')
###############################################################################

######################FEATURE SELECTION########################################
#####################DID NOT HELP, SO COMMENTING###############################
#Feature Selection using sklearn
#Uses the backward selection	
#Using feature selection, we rank the predictors as per their effect on the dependent variable
#We select the important variables and run our model only on those variables
#Import the RFE model and Logistic Regression
#from sklearn.feature_selection import RFE
##from sklearn.linear_model import LogisticRegression
##Model using Logistic Model
#model = LogisticRegression()
##Use the RFE function to find the variables rankings
#rfe = RFE(model,193)
#rfe = rfe.fit(train_features,train_labels)
#
##Store the rankings of the variables in array a1
##The variable that is important has a rank 1, others have random values
#import numpy as py
#a1 = py.zeros(shape=(192,1))
##print(rfe.support_)
#a1 = rfe.ranking_
#
##Store all important variables to a list
#l1 = list()
#for i in range(0,191):
#    if(a1[i] != 1): ##Important variables have rank 1
#        l1.append(i)
#
##Drop the columns that are not important bsed on RFE
#train_features = train_features.drop(train_features.columns[l1],axis=1)
#test_features = test_features.drop(test_features.columns[l1],axis=1)
###############################################################################


#####################################FEATURE SCALING###########################
## ML models work best when the input variables are normally distributed
## StandardScaler makes the data as Gaussian with mean 0 and unit variance
## scale the train data
#from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)

## scale the test data
scaler2 = StandardScaler().fit(test_features)
test_features = scaler2.transform(test_features)

## Normalize the train labels using LabelEncoder
## It converts the categorical variable to 1s and 0s
#from sklearn import preprocessing
encoder= preprocessing.LabelEncoder()
train_labels2 = encoder.fit_transform(train_labels)
###############################################################################

####################################MODELLING##################################

##########################1.NEURAL NETWORK#####################################
###################THE CHAMPION################################################
##Import the keras package for neural network
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical

##We need the train labels to be of categorical type to implement in neural network
train_labels_cat = to_categorical(train_labels2)

##Adding attributes of the neural network model
model = Sequential()
##Adding the number of independent varaiables
model.add(Dense(1024,input_dim=196))
##We are taking the network with 2 hidden layers
##First layer is using sigmoid function with threshold of 0.2
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))
model.add(Dense(512))
##Second layer is using sigmoid function with threshold of 0.3
model.add(Dropout(0.3))
model.add(Activation('sigmoid'))
model.add(Dense(99))
##The output layer is using softmax
model.add(Activation('softmax'))

##Compile chooses the best way to represnt the network
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

##Fit the model
nnfit = model.fit(train_features,train_labels_cat,batch_size=128,nb_epoch=100,verbose=0)
##Predict the test labels
test_Pred_Labels = model.predict_proba(test_features)
##Generate the output file
submit = pd.DataFrame(test_Pred_Labels ,index=x_testid , columns=encoder.classes_)
submit.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/submit_NNW.csv")

###############################################################################




##########################2.MULTINOMAIL LOGISTIC REGRESSION####################
###################SECOND BEST RESULTS#########################################
##Most important-Tuning of Logistic Regression Model parameters
##The GridSearchCV instance implements the usual estimator API: when “fitting” 
##it on a dataset all the possible combinations of parameter values are 
##evaluated and the best combination is retained.
#from sklearn.grid_search import GridSearchCV
#Set the tuning parameter
params = {'C':[1000,1200,1400], 'tol': [0.00001]}
#params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 'tol': [0.001, 0.0001, 0.005]}
##Create the Logistic Regression model
logreg = LogisticRegression(solver='newton-cg', multi_class='multinomial')
##Use the Grid Search with the params
clf = GridSearchCV(logreg,params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)
##Fit the models
clf_fit = clf.fit(train_features,train_labels2)
##Predict the test_labels
test_labels= clf.predict_proba(test_features)


##Generate the submission file
submit = pd.DataFrame(test_labels ,index=x_testid , columns=encoder.classes_)
submit.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/submit_LOgModel.csv")
###############################################################################

##########################3.RANDOM FOREST######################################
#Import the packages for Random Forest
from sklearn.ensemble import RandomForestClassifier
#Initialize the classifier
clf= RandomForestClassifier(n_estimators=1000)
#Fit the model
clffit = clf.fit(train_features,train_labels2)
#Predict the test labels
test_labels= clf.predict_proba(test_features)

#Generate the Submission FIle
submit = pd.DataFrame(test_labels ,index=x_testid , columns=encoder.classes_)
submit.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/submit_RAFModel.csv")
###############################################################################

#########################4.SVM#################################################
#Import the packages for SVM
from sklearn import svm
#Initialize the classifier
clf =svm.SVC(probability=True)
#Fit the model
clffit = clf.fit(train_features,train_labels2)
#Predict the test labels
test_labels= clf.predict_proba(test_features)

#Generate the Submission FIle
submit = pd.DataFrame(test_labels ,index=x_testid , columns=encoder.classes_)
submit.to_csv("C:/Users/parth/Desktop/MIA/MidTerm/submit_SVMModel.csv")
###############################################################################




