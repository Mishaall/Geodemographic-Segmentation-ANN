# Artificial Neural Network- deep learning used in computer vidion- recognising faces, medicine- making predictions for business problemsv, tumors in images
#predicting based on independat variables- classification problem 

# Installing Theano - open source numerical computations lib- based on numpy syntax runs on cpu
#cpu- main processor on comp with general purpose (gpu- graphic purposes ) : gpu muhc more powerful and runs more floating points [when forward propogatting/ back propogated, faster and better choice is gpu ]
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow- open source numerical lib cpu and gpu
#developed by google brain, mostly used for research and development- used to build NN from scratch
# pip install tensorflow
#pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
#VERY IMP pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed --upgrade \



# Installing Keras- wraps (based on/ runs on)theano and tensorflow
#good to build dl models with few lines of code- efficient dl models
# pip install --upgrade keras

#STEP 1: set working library
#tryning to figure out solution to classification program and output binary values
#STEP 2: take whole classification library except visulaisation because too many values
#STEP 3: import libs +dataset with new name
#dataset- customer took info and observed whether stayed or left in 6 months

# Part 1 - Data Preprocessing---------------------------------------------------------------------------------

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #matrix of features- look at index from data set - first 3 columns are independant/will not impact decision of customer --> customers with low credit score will have impact
                    #lower bound: upper bound- takes all indexes from 3:12
y = dataset.iloc[:, 13].values
                    #dependant variable starts @13

# Encoding categorical data- must so before splittign test and training
                    # use categorical template- have to encode [dependant varibale is categorical so dont have to label encode; but have categories that are strings so have to ]
                    # look at matrix of feature from console- only have 2 categorical ind variables- gender and country 
                    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # need to create 2

#for country 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # convert strings from france, spain, germany to 1 and 2 
labelencoder_X_2 = LabelEncoder()

#for gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #fit [num] indicates chqanged index
onehotencoder = OneHotEncoder(categorical_features = [1]) # need to create dummy variables 
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #remove one dummy variable, take all lines of matrix and all colums- take all columns except last index of last variable 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #change to test on 20000 variables

# Feature Scaling - compulsory to apply because of highly intensive computational requirements 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# all of data is preprocessed 

# Part 2 - Now let's make the ANNs--------------------------------------------------------------------------

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #sequential module- used to initialize 
from keras.layers import Dense #creates layer in ann

# Initialising the ANN -  making network with succesive layers- will initialise by defining as objetc of sequential clase
classifier = Sequential() #is nn model which is classifier
#dont need to put any argument in brackets because defining by layers below

###############################################################################
#NOTE: steps to training an ANN with stochastic gradient descent review
# 1. Randomly initialize weights to small nums close to zero but not zero 
# 2. input first observation of dataset in inpu tlayer, each feature in each node
# 3. forward propogation: from left to right, neurons activated in way that each neurons activation impacted by weight 
# 4. compare predicted to actual, measure error
# 5. back propogation- from left to right, error is back propogated. Update weights according to how much they are responsible for error
    #learning rate deicdes how muhc update wieghts
# 6. repeat steps 1 to 5 and update weights after each observation (reinforcement learning) or only update wieghts after a group (batch learning)
# 7. when whole training set passed through ANN, makes epoch, redo more

###############################################################################

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#dense function intitializes random wights
#num of nodes is num of nodes is num of ind vairables in matrix -11 
#deifne hidden layer using actviation function - rectifier
#argument is layers- using dense function 
#chose num of nodes in hidden layer as average of nodes in input layer and output 
# using rectifier actviation function for 


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# dont need input dim parameters


# Adding the output layer
classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))
#sigmoid fn ideal for output- get probabilities for different clasees even new obs of set
#getting probabilities thorugh segmentation function 
# if more than 2 categories, then add softmax - sigmoid with dependant variable with mroe than 2 categories

# Compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = [ 'accuracy' ])
#adam is sdg
#lossfunction


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model-------------------------------------------------------

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #threshold to predict whether will stay or not


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework Question

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

#Part 4 - Evaluating, Improving and Tuning the ANN---------------------------------------------------------

#Evaluating
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_clasifier():
    classifier = Sequential ()
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier  = KerasClassifier(build_fn = build_clasifier, batch_size = 10, epochs = 100)    
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed tp the hidden layers only

# Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_clasifier(optimizer):
    classifier = Sequential ()
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier  = KerasClassifier(build_fn = build_clasifier)
parameters = {'batch_size' : [25, 32],'epochs' : [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV (estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



