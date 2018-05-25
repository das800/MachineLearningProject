
# coding: utf-8

# Machine learning project

# Project Instructions:
# You can start with the classification project if it helps you. It is the file Classification.ipynb.
# During the classification lecture we did not talk about certain modeling practices that we used in the two auto case studies and that was intentional so that you can apply them in your IRIS Project. Your assignment is as follow....
# 1)Scale your independent variables (the x variables) and explain why we typically scale the variables. (20 points)Normalizing the data 
# 2)Produce a cross validation score with K=5 for the decision tree model like we did in the auto case studies and explain why we typically use cross validation in modeling (20 points)
# 3)Produce a confusion matrix for the decision tree model and explain what you conclude from the confusion matrix (20 points)
# 4)Produce a confusion matrix for the random forest model and explain what you conclude from the confusion matrix (20 points) 
# 5)This is the hardest part of the project because we did not go over the k selection but there is plenty of literature in the internet. In the classification lecture we used k-neighborhoud and we selected k=5. However, it may not be the optimum k value. Write a program that selects the optimum K for the K Neighbors Classifier model. (20 points)
# What I am looking for is the python code and the output of the code.
# A lot of what I am asking we we already cover in the last two lectures feel free to utilize the codes from the case studies. 
# We will go over the solution in the next lecture.
# IRIS data set is the most commonly used data set in modeling, that is why it comes with sklearn package. I recommend you become familiar with it as a lot of interview questions are based on modeling the IRIS data set.
# Regards,
# DP

# # Import libraries and data

# In[1]:


#Importing the packages 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Scale Iris Data (using Normalization)
# - Data can be scaled or preprocessed in 2 ways. Normalization scales the data by putting it in the range between 1 and 0. Standardization creates a distribution out of the data with mean 0 and unit variance. Normalization is good for Iris as it is better when working with k-nearest neighbors and regression coefficients. This is why I have chosen to normalize this data set

# In[2]:


#loading and scaling iris data set
iris = datasets.load_iris() 
irisX = iris.data[:,:4]      #independant var X
irisY = iris.target           #target var Y

#print(irisX)

irisX = preprocessing.normalize(irisX) # sklearn.preprocessing
#print(irisX)
data  = iris['data']
irisDataFrame = pd.DataFrame(data)
features = iris['feature_names']
header  = ['Sepal Length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal Width(cm)']
irisDataFrame.columns = header
irisDataFrame['Target'] = target = irisY
targetNames = list(iris.target_names)
lst = [targetNames[i].title() for i in irisDataFrame['Target'] ]
irisDataFrame['Target Name']  = lst


# # Logistic Regression Model

# In[3]:


# C specifies the Inverse of regularization strength. The smaller value the stronger the regularization.
# Fitting with a Logistic Regression model
iris_logr = LogisticRegression(C=1e4)
iris_logr.fit(irisX, irisY)
setosa_coeff, versicolor_coeff, virginica_coeff = iris_logr.coef_
setosa_interc, versicolor_interc, virginica_interc = iris_logr.intercept_

# Observe the coefficients of the predictors
print("Coefficents of setosa, versicolor, virginica: %s" %str(iris_logr.coef_))

#Observer the intercept
print("Intercept: %s" %iris_logr.intercept_)

#Model Score
## determination
print("The coefficient of determination for multi-class logistic regression model is: %.4f" %iris_logr.score(irisX, irisY))


# In[4]:


from sklearn.tree import DecisionTreeClassifier
iris_dtree = DecisionTreeClassifier()
iris_dtree.fit(irisX,irisY)
#Model Score
## determination
print("The coefficient of determination for the Decision Tree model is: %.4f" %iris_dtree.score(irisX, irisY))


# # Random Forest Classifier 

# In[5]:


from sklearn.ensemble import RandomForestClassifier

iris_rf = RandomForestClassifier(random_state=1)
iris_rf.fit(irisX,irisY)
#Model Score
print("The coefficient of determination for the Random Forest model is: %.4f" %iris_rf.score(irisX, irisY))


# # K- Fold Cross Validation

# In[6]:


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

x = irisX
y = irisY
kf = KFold(n_splits=5, random_state=None, shuffle=True)
kf.get_n_splits(x)
for train_i, test_i in kf.split(x):
    print("TRAIN:", train_i, "TEST:", test_i)
    X_train, X_test = x[train_i], x[test_i]
    y_train, y_test = y[train_i], y[test_i]


# # 2. KFold Score
# We use cross validation so as to better predict the test error and gauge the accuracy of our model by using such a prediction. it is used over a validation set so as to not decrease the size of our training data too much as that raises error.

# In[7]:


#K- Fold Score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

iris_dtree.fit(X_train, y_train)
y_pred = cross_val_predict(iris_dtree,irisX,irisY,cv=5)
accuracies_cv_new = cross_val_score(iris_dtree, X_train, y_train, cv=5)
print('The 5-fold cross validation accuracy of this model is: %s' % '{0:.3%}'.format(accuracies_cv_new.mean()))
print('The accuracy of this model is: %s' % '{0:.3%}'.format(metrics.accuracy_score(irisY,y_pred)))


# # 3. Confusion Matrix for Decision Tree
# This is the confusion matrix to guage error rate. (code copied from lectures)

# In[11]:


#Confusion Matrix,
from sklearn.metrics import classification_report,confusion_matrix,auc,accuracy_score
from mlxtend.plotting import plot_confusion_matrix  
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


y_pred = cross_val_predict(iris_dtree,irisX,irisY,cv=5)
cm = confusion_matrix(irisY,y_pred)
fig = plot_confusion_matrix(conf_mat=cm,figsize=(4,4),cmap=plt.cm.Reds,hide_spines=True)
plt.title('Confusion Matrix',fontsize=14)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
# Turn the axis grid off
plt.grid('off')

# Print out metrics
print('The 5-fold cross validation accuracy of this model is: %s' % '{0:.3%}'.format(accuracies_cv_new.mean()))
print('The accuracy of this model is: %s' % '{0:.3%}'.format(accuracy_score(irisY,y_pred)))


# The matrix shows us that the decision tree is a very good predictor for the Iris set. this is apparent from the heavy principal diagonal and 6/150 classification error.

# # 4. Confusion Matrix for Random Forest Tree
# This is the confusion matrix to guage error rate. (code copied from lectures)

# In[9]:


#K- Fold Score Random Forest Model
iris_rf.fit(X_train, y_train)


y_pred_rf = cross_val_predict(iris_rf,irisX,irisY,cv=5)
conf_mat_rf = confusion_matrix(irisY,y_pred_rf)
accuracies_cv_new = cross_val_score(iris_rf, X_train, y_train, cv=5)


fig = plot_confusion_matrix(conf_mat=conf_mat_rf,figsize=(4,4),cmap=plt.cm.Reds,hide_spines=True)
plt.title('Confusion Matrix',fontsize=14)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
# Turn the axis grid off
plt.grid('off')

# Print out metrics
print('The 5-fold cross validation accuracy of this model is: %s' % '{0:.3%}'.format(accuracies_cv_new.mean()))
print('The accuracy of this model is: %s' % '{0:.3%}'.format(accuracy_score(irisY,y_pred)))


# The matrix shows us that the decision tree is a very good predictor for the Iris set. this is apparent from the heavy principal diagonal and 4/150 classification error.

# One thing to note is that while are very good models (as shown by their accuracy and confusion scores) the Decision Tree has a slightly lower KFold score and more errors in the confusion matrix. Despite this, decision trees suite the iris data set better.

# # 5. Optimum K for K Neighbors model
# The best K values were assertained by looking at the highest 10-Fold cross validation accuracy of K-neighbor models with K ranging from 1 to 50.

# In[10]:


from sklearn.neighbors import KNeighborsClassifier 

def KFoldScore():
    KScore = []
    for i in range(1, 51):  
        KNeighbors = KNeighborsClassifier(n_neighbors=i)
        KNeighbors.fit(X_train, y_train)
        #from lecture
        accuracies_cv_new = cross_val_score(KNeighbors, X_train, y_train, cv=10)
        kscore = accuracies_cv_new.mean()
        KScore.append(kscore)
    return (KScore)

def getMax():
    KScores = KFoldScore()
    maxKScore = max(KScores)
    bestK = []
    for i in range(1, 51):
        if (KScores[i-1] == maxKScore):
            bestK.append(i)
    return (bestK, maxKScore)

#print(KFoldScore())
#print(getMax())

kresult = getMax()
print("The best K values < 50 are the following (score =", kresult[1],"):", end=" ")
for i in kresult[0]:
    print(i, end=",")


